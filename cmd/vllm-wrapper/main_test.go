package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"os"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"
)

func TestWakeUpVLLM(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/wake_up" {
			t.Errorf("Unexpected request: %s %s", r.Method, r.URL.Path)
		}
		if r.URL.Query().Get("tags") != "" {
			t.Errorf("Level-1 wake should not include tags param, got: %s", r.URL.Query().Get("tags"))
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer ts.Close()

	if err := wakeUpVLLM(ts.URL, 1); err != nil {
		t.Fatalf("wakeUpVLLM failed: %v", err)
	}

	ts2 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/wake_up" {
			t.Errorf("Unexpected request: %s %s", r.Method, r.URL.Path)
		}
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer ts2.Close()

	if err := wakeUpVLLM(ts2.URL, 1); err == nil {
		t.Errorf("wakeUpVLLM expected error for non-200 response")
	}
}

func TestVllmWrapper_WakeUpLevel2(t *testing.T) {
	var (
		mu       sync.Mutex
		requests []struct {
			method string
			path   string
			query  string
			body   string
		}
	)

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		buf := make([]byte, 4096)
		n, _ := r.Body.Read(buf)
		_ = r.Body.Close()
		body := string(buf[:n])

		mu.Lock()
		requests = append(requests, struct {
			method string
			path   string
			query  string
			body   string
		}{
			method: r.Method,
			path:   r.URL.Path,
			query:  r.URL.RawQuery,
			body:   body,
		})
		mu.Unlock()

		w.WriteHeader(http.StatusOK)
	}))
	defer ts.Close()

	err := wakeUpVLLM(ts.URL, 2)
	if err != nil {
		t.Fatalf("wakeUpVLLM(url, 2) failed: %v", err)
	}

	mu.Lock()
	reqs := requests
	mu.Unlock()

	if len(reqs) != 4 {
		t.Fatalf("Expected 4 requests, got %d", len(reqs))
	}

	if reqs[0].method != http.MethodPost || reqs[0].path != "/wake_up" || reqs[0].query != "tags=weights" || strings.TrimSpace(reqs[0].body) != "" {
		t.Errorf("Step 1 invalid: %s %s?%s body=%q", reqs[0].method, reqs[0].path, reqs[0].query, reqs[0].body)
	}

	if reqs[1].method != http.MethodPost || reqs[1].path != "/collective_rpc" || !strings.Contains(reqs[1].body, `"method":"reload_weights"`) {
		t.Errorf("Step 2 invalid: %s %s body=%q", reqs[1].method, reqs[1].path, reqs[1].body)
	}

	if reqs[2].method != http.MethodPost || reqs[2].path != "/wake_up" || reqs[2].query != "tags=kv_cache" || strings.TrimSpace(reqs[2].body) != "" {
		t.Errorf("Step 3 invalid: %s %s?%s body=%q", reqs[2].method, reqs[2].path, reqs[2].query, reqs[2].body)
	}

	if reqs[3].method != http.MethodPost || reqs[3].path != "/reset_prefix_cache" || strings.TrimSpace(reqs[3].body) != "" {
		t.Errorf("Step 4 invalid: %s %s body=%q", reqs[3].method, reqs[3].path, reqs[3].body)
	}
}

func TestVllmWrapper_WakeUpLevel2FailFast(t *testing.T) {
	var (
		mu       sync.Mutex
		requests []struct {
			method string
			path   string
			query  string
		}
	)

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = r.Body.Close()

		mu.Lock()
		requests = append(requests, struct {
			method string
			path   string
			query  string
		}{
			method: r.Method,
			path:   r.URL.Path,
			query:  r.URL.RawQuery,
		})
		mu.Unlock()

		if r.URL.Path == "/collective_rpc" {
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer ts.Close()

	err := wakeUpVLLM(ts.URL, 2)
	if err == nil {
		t.Fatalf("Expected error on step 2 failure, got nil")
	}

	mu.Lock()
	reqs := requests
	mu.Unlock()

	// step 1 (1 req) + step 2 retries (5 attempts) = 6 total
	if len(reqs) != 6 {
		t.Fatalf("Expected 6 requests (1 + 5 retries), got %d", len(reqs))
	}

	if reqs[0].path != "/wake_up" || reqs[0].query != "tags=weights" {
		t.Errorf("Step 1 unexpected: %s %s", reqs[0].path, reqs[0].query)
	}
	if reqs[1].path != "/collective_rpc" {
		t.Errorf("Step 2 unexpected: %s", reqs[1].path)
	}
}

func TestVllmWrapper_WakeUpLevel2ResetPrefixCacheWarning(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = r.Body.Close()
		path := r.URL.Path

		switch path {
		case "/wake_up", "/collective_rpc":
			w.WriteHeader(http.StatusOK)
		case "/reset_prefix_cache":
			w.WriteHeader(http.StatusInternalServerError)
		default:
			t.Errorf("Unexpected path: %s", path)
			w.WriteHeader(http.StatusOK)
		}
	}))
	defer ts.Close()

	err := wakeUpVLLM(ts.URL, 2)
	if err != nil {
		t.Fatalf("wakeUpVLLM should not return error for step 4 failure, got: %v", err)
	}
}

func TestVllmWrapper_SleepQueryParams(t *testing.T) {
	tests := []struct {
		name     string
		level    int
		mode     string
		expected string
	}{
		{"level-2-wait", 2, "wait", "level=2&mode=wait"},
		{"level-2-abort", 2, "abort", "level=2&mode=abort"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method != http.MethodPost || r.URL.Path != "/sleep" {
					t.Errorf("Expected POST /sleep, got %s %s", r.Method, r.URL.Path)
				}
				buf := make([]byte, 4096)
				n, _ := r.Body.Read(buf)
				_ = r.Body.Close()
				body := strings.TrimSpace(string(buf[:n]))
				if body != "" && strings.HasPrefix(body, "{") {
					t.Errorf("Expected no JSON body, got %q", body)
				}
				if r.URL.RawQuery != tt.expected {
					t.Errorf("Expected query %q, got %q", tt.expected, r.URL.RawQuery)
				}
				w.WriteHeader(http.StatusOK)
			}))
			defer ts.Close()

			err := sendSleepRequest(ts.URL, tt.level, tt.mode)
			if err != nil {
				t.Fatalf("sendSleepRequest failed: %v", err)
			}
		})
	}
}

func TestWaitForHealthy(t *testing.T) {
	// Test successful health check
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			t.Errorf("Unexpected path: %s", r.URL.Path)
		}
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"data":[]}`))
	}))
	defer ts.Close()

	if err := waitForHealthyWithPath(ts.URL, "/v1/models", 2*time.Second); err != nil {
		t.Fatalf("waitForHealthy failed: %v", err)
	}

	ts2 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(2 * time.Second)
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"data":[]}`))
	}))
	defer ts2.Close()

	err := waitForHealthyWithPath(ts2.URL, "/v1/models", 1*time.Second)
	if err == nil {
		t.Errorf("waitForHealthy expected timeout error")
		return
	}
	if err != context.DeadlineExceeded {
		t.Errorf("waitForHealthy expected context deadline exceeded, got %v", err)
	}
}

func TestStartDaemon(t *testing.T) {
	pid, err := startDaemon("true", "http://127.0.0.1:12345/health", "/health", 10*time.Millisecond)
	if err == nil {
		t.Fatalf("startDaemon expected error but got nil")
	}
	if pid != 0 {
		t.Errorf("startDaemon expected pid=0 on failure, got %d", pid)
	}
	if !strings.Contains(err.Error(), "daemon did not become healthy") {
		t.Errorf("error expected to contain 'daemon did not become healthy', got %v", err)
	}
}

func TestStartDaemon_SurvivesParentExit(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer ts.Close()

	pid, err := startDaemon("sleep 30", ts.URL, "/", 5*time.Second)
	if err != nil {
		t.Fatalf("startDaemon failed: %v", err)
	}
	if pid == 0 {
		t.Fatal("expected non-zero pid")
	}

	if daemonLogStop != nil {
		daemonLogStop()
		daemonLogStop = nil
	}

	proc, err := os.FindProcess(pid)
	if err != nil {
		t.Fatalf("FindProcess(%d) failed: %v", pid, err)
	}
	// Signal 0 checks process liveness without delivering a real signal
	if err := proc.Signal(syscall.Signal(0)); err != nil {
		t.Errorf("daemon process %d died after tail stop: %v", pid, err)
	}

	proc.Kill()
}

func TestTailFile(t *testing.T) {
	tmpFile, err := os.CreateTemp(t.TempDir(), "tailtest")
	if err != nil {
		t.Fatal(err)
	}

	stop := tailFile(tmpFile)
	defer stop()

	tmpFile.WriteString("line one\n")
	tmpFile.WriteString("line two\n")
	tmpFile.Sync()

	time.Sleep(250 * time.Millisecond)
	stop()
}

func TestVllmWrapper_InjectMetricsParams(t *testing.T) {
	t.Run("non-streaming", func(t *testing.T) {
		body := `{"model":"test","messages":[{"role":"user","content":"hi"}]}`
		r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
		r.Header.Set("Content-Type", "application/json")

		injectMetricsParams(r)

		result, _ := io.ReadAll(r.Body)
		var data map[string]any
		if err := json.Unmarshal(result, &data); err != nil {
			t.Fatalf("failed to parse modified body: %v", err)
		}

		if data["include_metrics"] != true {
			t.Errorf("include_metrics = %v, want true", data["include_metrics"])
		}
		if data["stream_options"] != nil {
			t.Errorf("stream_options should not be set for non-streaming request")
		}
	})

	t.Run("streaming", func(t *testing.T) {
		body := `{"model":"test","messages":[],"stream":true}`
		r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))

		injectMetricsParams(r)

		result, _ := io.ReadAll(r.Body)
		var data map[string]any
		if err := json.Unmarshal(result, &data); err != nil {
			t.Fatalf("failed to parse modified body: %v", err)
		}

		if data["include_metrics"] != true {
			t.Errorf("include_metrics = %v, want true", data["include_metrics"])
		}
		opts, ok := data["stream_options"].(map[string]any)
		if !ok {
			t.Fatalf("stream_options not set for streaming request")
		}
		if opts["include_usage"] != true {
			t.Errorf("stream_options.include_usage = %v, want true", opts["include_usage"])
		}
	})

	t.Run("streaming with existing stream_options", func(t *testing.T) {
		body := `{"model":"test","messages":[],"stream":true,"stream_options":{"continuous_usage_stats":true}}`
		r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))

		injectMetricsParams(r)

		result, _ := io.ReadAll(r.Body)
		var data map[string]any
		if err := json.Unmarshal(result, &data); err != nil {
			t.Fatalf("failed to parse modified body: %v", err)
		}

		opts := data["stream_options"].(map[string]any)
		if opts["include_usage"] != true {
			t.Errorf("stream_options.include_usage = %v, want true", opts["include_usage"])
		}
		if opts["continuous_usage_stats"] != true {
			t.Errorf("existing stream_options.continuous_usage_stats was clobbered")
		}
	})

	t.Run("invalid JSON passes through unchanged", func(t *testing.T) {
		original := `not valid json`
		r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(original))

		injectMetricsParams(r)

		result, _ := io.ReadAll(r.Body)
		if string(result) != original {
			t.Errorf("body = %q, want unchanged %q", result, original)
		}
	})

	t.Run("nil body", func(t *testing.T) {
		r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
		r.Body = nil

		injectMetricsParams(r)
	})

	t.Run("content-length updated", func(t *testing.T) {
		body := `{"model":"x","messages":[]}`
		r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))

		injectMetricsParams(r)

		result, _ := io.ReadAll(r.Body)
		if int64(len(result)) != r.ContentLength {
			t.Errorf("ContentLength = %d, body len = %d", r.ContentLength, len(result))
		}
	})
}

func TestVllmWrapper_IsInferencePath(t *testing.T) {
	tests := []struct {
		path string
		want bool
	}{
		{"/v1/chat/completions", true},
		{"/v1/completions", true},
		{"/v1/models", false},
		{"/health", false},
		{"/metrics", false},
		{"/v1/embeddings", false},
	}
	for _, tt := range tests {
		if got := isInferencePath(tt.path); got != tt.want {
			t.Errorf("isInferencePath(%q) = %v, want %v", tt.path, got, tt.want)
		}
	}
}

func TestVllmWrapper_ProxyInjectsMetrics(t *testing.T) {
	var receivedBody []byte
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":10}}`))
	}))
	defer upstream.Close()

	proxyURL, _ := url.Parse(upstream.URL)
	proxy := httputil.NewSingleHostReverseProxy(proxyURL)

	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost && isInferencePath(r.URL.Path) {
			injectMetricsParams(r)
		}
		proxy.ServeHTTP(w, r)
	})

	reqBody := `{"model":"test","messages":[{"role":"user","content":"hello"}],"stream":true}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}

	var data map[string]any
	if err := json.Unmarshal(receivedBody, &data); err != nil {
		t.Fatalf("upstream received invalid JSON: %v\nbody: %s", err, receivedBody)
	}
	if data["include_metrics"] != true {
		t.Errorf("upstream did not receive include_metrics=true")
	}
	opts, ok := data["stream_options"].(map[string]any)
	if !ok || opts["include_usage"] != true {
		t.Errorf("upstream did not receive stream_options.include_usage=true, got %v", data["stream_options"])
	}
}

func TestVllmWrapper_ProxySkipsNonInferencePaths(t *testing.T) {
	var receivedBody []byte
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedBody, _ = io.ReadAll(r.Body)
		w.WriteHeader(http.StatusOK)
	}))
	defer upstream.Close()

	proxyURL, _ := url.Parse(upstream.URL)
	proxy := httputil.NewSingleHostReverseProxy(proxyURL)

	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost && isInferencePath(r.URL.Path) {
			injectMetricsParams(r)
		}
		proxy.ServeHTTP(w, r)
	})

	original := `{"query":"test"}`
	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", strings.NewReader(original))
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if !bytes.Equal(receivedBody, []byte(original)) {
		t.Errorf("non-inference path body was modified: got %q, want %q", receivedBody, original)
	}
}
