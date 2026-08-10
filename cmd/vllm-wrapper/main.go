package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

const httpTimeout = 30 * time.Second

var httpClient = &http.Client{Timeout: httpTimeout}

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "Usage: %s <command> [args]\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "Commands:\n")
		fmt.Fprintf(os.Stderr, "  serve    Start as a forward proxy (for cmd)\n")
		fmt.Fprintf(os.Stderr, "  sleep    Put vLLM to sleep (for cmdStop)\n")
		os.Exit(1)
	}

	switch os.Args[1] {
	case "serve":
		serveCmd(os.Args[2:])
	case "sleep":
		sleepCmd(os.Args[2:])
	default:
		fmt.Fprintf(os.Stderr, "Unknown command: %s\n", os.Args[1])
		os.Exit(1)
	}
}

func serveCmd(args []string) {
	var (
		vllmURL     string
		listenAddr  string
		sleepLevel  int
		startCmd    string
		healthPath  string
		waitTimeout time.Duration
	)
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	fs.StringVar(&vllmURL, "vllm-url", "", "Base URL of vLLM server (e.g., http://127.0.0.1:8000)")
	fs.StringVar(&listenAddr, "listen", "", "Address to listen on (e.g., :$PORT)")
	fs.IntVar(&sleepLevel, "sleep-level", 1, "Sleep level to use when sleeping (default 1)")
	fs.StringVar(&startCmd, "start-cmd", "", "Command to start the vLLM daemon if not running (e.g., 'docker run ...')")
	fs.StringVar(&healthPath, "health-path", "/health", "Health check path (default /health)")
	fs.DurationVar(&waitTimeout, "wait-timeout", 120*time.Second, "Timeout waiting for daemon to become healthy")
	fs.Parse(args)

	if vllmURL == "" {
		log.Fatalf("--vllm-url is required")
	}
	if listenAddr == "" {
		log.Fatalf("--listen is required")
	}
	if startCmd == "" {
		log.Fatalf("--start-cmd is required")
	}

	vllmURL = strings.TrimRight(vllmURL, "/")

	var daemonPID int

	if err := checkHealthy(vllmURL, healthPath); err != nil {
		log.Printf("vLLM daemon not reachable (%v), attempting to wake up", err)
		if err := wakeUpVLLM(vllmURL, sleepLevel); err != nil {
			log.Printf("Wake up failed: %v, attempting to start daemon", err)
			daemonPID, err = startDaemon(startCmd, vllmURL, healthPath, waitTimeout)
			if err != nil {
				log.Fatalf("Failed to start daemon: %v", err)
			}
		} else {
			log.Printf("Wake up sent, waiting for healthy state")
			if err := waitForHealthyWithPath(vllmURL, healthPath, waitTimeout); err != nil {
				log.Fatalf("vLLM health check failed after wake up: %v", err)
			}
		}
	} else {
		log.Printf("vLLM daemon is reachable at %s%s, attempting to wake up if asleep", vllmURL, healthPath)
		if err := wakeUpVLLM(vllmURL, sleepLevel); err != nil {
			if sleepLevel >= 2 {
				log.Fatalf("vLLM level-%d wake failed on reachable daemon: %v", sleepLevel, err)
			}
			log.Printf("Warning: wake up failed (but continuing): %v", err)
		}
		log.Printf("Waiting for vLLM to be healthy after wake up")
		healthTimeout := 10 * time.Second
		if sleepLevel >= 2 {
			healthTimeout = waitTimeout
		}
		if err := waitForHealthyWithPath(vllmURL, healthPath, healthTimeout); err != nil {
			log.Fatalf("vLLM health check failed after wake up: %v", err)
		}
	}

	proxyURL, err := url.Parse(vllmURL)
	if err != nil {
		log.Fatalf("Invalid vLLM URL %q: %v", vllmURL, err)
	}
	proxy := httputil.NewSingleHostReverseProxy(proxyURL)

	transport := &http.Transport{
		Proxy: http.ProxyFromEnvironment,
		DialContext: (&net.Dialer{
			Timeout:   30 * time.Second,
			KeepAlive: 30 * time.Second,
		}).DialContext,
		TLSHandshakeTimeout:   10 * time.Second,
		ResponseHeaderTimeout: 30 * time.Second,
		ExpectContinueTimeout: 1 * time.Second,
		MaxIdleConns:          100,
		MaxIdleConnsPerHost:   10,
		IdleConnTimeout:       90 * time.Second,
	}
	proxy.Transport = transport

	proxy.ModifyResponse = func(resp *http.Response) error {
		if strings.Contains(strings.ToLower(resp.Header.Get("Content-Type")), "text/event-stream") {
			resp.Header.Set("X-Accel-Buffering", "no")
		}
		return nil
	}

	srv := &http.Server{
		Addr: listenAddr,
		Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method == http.MethodPost && isInferencePath(r.URL.Path) {
				injectMetricsParams(r)
			}
			proxy.ServeHTTP(w, r)
		}),
	}

	go func() {
		log.Printf("Starting vllm-wrapper serve on %s -> %s", listenAddr, vllmURL)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("ListenAndServe: %v", err)
		}
	}()

	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM, syscall.SIGQUIT)
	sig := <-c

	if sig == syscall.SIGQUIT && daemonPID > 0 {
		log.Printf("SIGQUIT received, killing vLLM daemon process group (pid %d)", daemonPID)
		if err := syscall.Kill(-daemonPID, syscall.SIGKILL); err != nil {
			log.Printf("process group kill failed, killing pid directly: %v", err)
			syscall.Kill(daemonPID, syscall.SIGKILL)
		}
	}

	if daemonLogStop != nil {
		daemonLogStop()
	}

	log.Println("Shutting down vllm-wrapper serve...")
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := srv.Shutdown(ctx); err != nil {
		log.Fatalf("Server shutdown failed: %v", err)
	}
	log.Println("Server stopped")
}

func sleepCmd(args []string) {
	var (
		vllmURL    string
		sleepLevel int
		sleepMode  string
		servePID   int
	)
	fs := flag.NewFlagSet("sleep", flag.ExitOnError)
	fs.StringVar(&vllmURL, "vllm-url", "", "Base URL of vLLM server (e.g., http://127.0.0.1:8000)")
	fs.IntVar(&sleepLevel, "sleep-level", 1, "Sleep level to use (default 1)")
	fs.StringVar(&sleepMode, "sleep-mode", "wait", "Sleep mode: 'wait' (drain requests) or 'abort' (immediate)")
	fs.IntVar(&servePID, "pid", 0, "PID of the serve process to signal after sleep succeeds (optional)")
	fs.Parse(args)

	if vllmURL == "" {
		log.Fatalf("--vllm-url is required")
	}
	if sleepMode != "wait" && sleepMode != "abort" {
		log.Fatalf("--sleep-mode must be 'wait' or 'abort', got: %q", sleepMode)
	}
	vllmURL = strings.TrimRight(vllmURL, "/")

	if err := sendSleepRequest(vllmURL, sleepLevel, sleepMode); err != nil {
		log.Fatalf("Failed to send sleep request: %v", err)
	}

	log.Printf("Successfully put vLLM to sleep (level %d)", sleepLevel)

	if servePID > 0 {
		if err := syscall.Kill(servePID, syscall.SIGTERM); err != nil {
			log.Printf("Warning: failed to signal serve process (pid %d): %v", servePID, err)
		} else {
			log.Printf("Sent SIGTERM to serve process (pid %d)", servePID)
		}
	}
}

func sendSleepRequest(vllmURL string, level int, mode string) error {
	reqURL := vllmURL + "/sleep?level=" + strconv.Itoa(level) + "&mode=" + mode
	req, err := http.NewRequest(http.MethodPost, reqURL, nil)
	if err != nil {
		return fmt.Errorf("failed to create sleep request: %w", err)
	}

	resp, err := httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to send sleep request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("vLLM sleep request failed with status %d: %s", resp.StatusCode, resp.Status)
	}
	return nil
}

func wakeUpVLLM(vllmURL string, sleepLevel int) error {
	log.Printf("Waking up vLLM (sleep level %d)", sleepLevel)
	if sleepLevel == 1 {
		return wakeUpVLLMLvl1(vllmURL)
	}

	return wakeUpVLLMLvl2(vllmURL)
}

func wakeUpVLLMLvl1(vllmURL string) error {
	req, err := http.NewRequest(http.MethodPost, vllmURL+"/wake_up", strings.NewReader(""))
	if err != nil {
		return fmt.Errorf("failed to create /wake_up request: %w", err)
	}

	resp, err := httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to POST /wake_up: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusNoContent {
		return fmt.Errorf("/wake_up returned unexpected status %d: %s", resp.StatusCode, resp.Status)
	}
	return nil
}

func wakeUpVLLMLvl2(vllmURL string) error {
	log.Printf("Level-2 wake: step 1/4 - POST /wake_up?tags=weights")
	if err := doPostCheck(vllmURL+"/wake_up?tags=weights", http.MethodPost, nil, []int{http.StatusOK, http.StatusNoContent}); err != nil {
		return fmt.Errorf("step 1 (wake_up weights): %w", err)
	}
	log.Printf("Level-2 wake: step 1/4 complete")

	// Step 2: reload_weights via collective_rpc. The weights pool allocation
	// from step 1 may be async, so retry a few times with backoff.
	log.Printf("Level-2 wake: step 2/4 - POST /collective_rpc (reload_weights)")
	var lastErr error
	for attempt := 0; attempt < 5; attempt++ {
		if attempt > 0 {
			time.Sleep(time.Duration(attempt) * time.Second)
		}
		bodyStr := `{"method":"reload_weights"}`
		lastErr = doPostCheck(vllmURL+"/collective_rpc", http.MethodPost, strings.NewReader(bodyStr), []int{http.StatusOK})
		if lastErr == nil {
			break
		}
		log.Printf("Level-2 wake: step 2/4 attempt %d failed: %v, retrying...", attempt+1, lastErr)
	}
	if lastErr != nil {
		return fmt.Errorf("step 2 (collective_rpc): %w", lastErr)
	}
	log.Printf("Level-2 wake: step 2/4 complete")

	log.Printf("Level-2 wake: step 3/4 - POST /wake_up?tags=kv_cache")
	if err := doPostCheck(vllmURL+"/wake_up?tags=kv_cache", http.MethodPost, nil, []int{http.StatusOK, http.StatusNoContent}); err != nil {
		return fmt.Errorf("step 3 (wake_up kv_cache): %w", err)
	}
	log.Printf("Level-2 wake: step 3/4 complete")

	log.Printf("Level-2 wake: step 4/4 - POST /reset_prefix_cache")
	req, err := http.NewRequest(http.MethodPost, vllmURL+"/reset_prefix_cache", nil)
	if err != nil {
		log.Printf("Warning: step 4: failed to create /reset_prefix_cache request: %v", err)
		return nil
	}
	resp, err := httpClient.Do(req)
	if err != nil {
		log.Printf("Warning: step 4: failed to POST /reset_prefix_cache: %v", err)
		return nil
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		log.Printf("Warning: step 4: /reset_prefix_cache returned status %d: %s", resp.StatusCode, resp.Status)
	} else {
		log.Printf("Level-2 wake: step 4/4 complete")
	}
	return nil
}

func doPostCheck(urlStr, method string, body interface{}, acceptedStatuses []int) error {
	var reader interface {
		Read(p []byte) (n int, err error)
	}
	if body != nil {
		reader = body.(interface {
			Read(p []byte) (n int, err error)
		})
	}

	req, err := http.NewRequest(method, urlStr, reader)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}

	resp, err := httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	for _, status := range acceptedStatuses {
		if resp.StatusCode == status {
			return nil
		}
	}
	return fmt.Errorf("unexpected status %d: %s", resp.StatusCode, resp.Status)
}

func waitForHealthyWithPath(vllmURL string, healthPath string, timeout time.Duration) error {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, vllmURL+healthPath, nil)
		if err != nil {
			return err
		}
		resp, err := httpClient.Do(req)
		if err != nil {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			time.Sleep(1 * time.Second)
			continue
		}
		if resp.StatusCode == http.StatusOK {
			resp.Body.Close()
			return nil
		}
		resp.Body.Close()
		time.Sleep(1 * time.Second)
	}
	return ctx.Err()
}

func checkHealthy(vllmURL string, healthPath string) error {
	req, err := http.NewRequest(http.MethodGet, vllmURL+healthPath, nil)
	if err != nil {
		return err
	}
	resp, err := httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}
	return nil
}

func startDaemon(startCmd string, vllmURL string, healthPath string, waitTimeout time.Duration) (int, error) {
	logFile, err := openDaemonLogFile()
	if err != nil {
		return 0, fmt.Errorf("failed to create daemon log file: %w", err)
	}

	cmd := exec.Command("sh", "-c", startCmd)
	cmd.Stdout = logFile
	cmd.Stderr = logFile
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
	if err := cmd.Start(); err != nil {
		logFile.Close()
		return 0, fmt.Errorf("failed to start daemon command: %w", err)
	}

	log.Printf("Started daemon with PID %d, waiting for healthy state", cmd.Process.Pid)

	stopTail := tailFile(logFile)

	err = waitForHealthyWithPath(vllmURL, healthPath, waitTimeout)
	if err != nil {
		stopTail()
		_ = cmd.Process.Kill()
		return 0, fmt.Errorf("daemon did not become healthy: %w", err)
	}

	daemonLogStop = stopTail

	return cmd.Process.Pid, nil
}

var daemonLogStop func()

func openDaemonLogFile() (*os.File, error) {
	logDir := filepath.Join(os.TempDir(), "vllm-wrapper")
	if err := os.MkdirAll(logDir, 0755); err != nil {
		return nil, err
	}
	logPath := filepath.Join(logDir, "daemon.log")
	f, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return nil, err
	}
	log.Printf("Daemon log file: %s", logPath)
	return f, nil
}

// tailFile opens a read-only handle on the same log file and streams new
// lines to os.Stdout. A separate fd is needed because the write handle is
// held by the daemon process. We seek to end so only output produced after
// this invocation is printed.
func tailFile(f *os.File) func() {
	rf, err := os.Open(f.Name())
	if err != nil {
		log.Printf("Warning: cannot tail daemon log: %v", err)
		return func() {}
	}

	rf.Seek(0, io.SeekEnd)

	done := make(chan struct{})
	go func() {
		defer rf.Close()
		reader := bufio.NewReader(rf)
		for {
			select {
			case <-done:
				for {
					line, err := reader.ReadString('\n')
					if line != "" {
						fmt.Fprint(os.Stdout, line)
					}
					if err != nil {
						break
					}
				}
				return
			default:
			}

			line, err := reader.ReadString('\n')
			if line != "" {
				fmt.Fprint(os.Stdout, line)
			}
			if err != nil {
				time.Sleep(100 * time.Millisecond)
			}
		}
	}()

	var once sync.Once
	return func() {
		once.Do(func() { close(done) })
	}
}

// inferencePaths lists the vLLM endpoints that support per-request metrics.
var inferencePaths = []string{
	"/v1/chat/completions",
	"/v1/completions",
}

func isInferencePath(path string) bool {
	for _, p := range inferencePaths {
		if path == p {
			return true
		}
	}
	return false
}

// injectMetricsParams sets "include_metrics": true in the JSON request body
// so vLLM returns per-request timing metrics. For streaming requests it also
// sets stream_options.include_usage so the final chunk carries usage+metrics.
func injectMetricsParams(r *http.Request) {
	if r.Body == nil {
		return
	}

	body, err := io.ReadAll(r.Body)
	r.Body.Close()
	if err != nil || len(body) == 0 {
		r.Body = io.NopCloser(bytes.NewReader(body))
		return
	}

	var data map[string]any
	if err := json.Unmarshal(body, &data); err != nil {
		r.Body = io.NopCloser(bytes.NewReader(body))
		return
	}

	data["include_metrics"] = true

	if stream, ok := data["stream"].(bool); ok && stream {
		opts, _ := data["stream_options"].(map[string]any)
		if opts == nil {
			opts = make(map[string]any)
		}
		opts["include_usage"] = true
		data["stream_options"] = opts
	}

	modified, err := json.Marshal(data)
	if err != nil {
		r.Body = io.NopCloser(bytes.NewReader(body))
		return
	}

	r.Body = io.NopCloser(bytes.NewReader(modified))
	r.ContentLength = int64(len(modified))
}
