# vllm-wrapper

`vllm-wrapper` is a standalone helper program designed to be used as a model's `cmd` and `cmdStop` in llama-swap configurations for vLLM servers that have been started with `--enable-sleep-mode`.

It provides two subcommands:

- `serve`: Used as a model's `cmd`. Manages the vLLM daemon lifecycle: if the daemon is not running, starts it using the provided start command; if running but asleep, wakes it up; waits for it to be healthy, then runs a reverse proxy from a local port to the vLLM upstream.
- `sleep`: Used as a model's `cmdStop`. Sends a sleep request to the vLLM daemon to free VRAM while keeping the process alive.

## Why use this?

When using vLLM with llama-swap, you can leverage vLLM's sleep mode to drastically reduce swap-in times. Instead of stopping and starting the vLLM process (which incurs a cold start), you can put the vLLM daemon to sleep when not in use (via `cmdStop`) and wake it up when needed (via `cmd`). This keeps the vLLM process running, preserving the GPU context and allowing for near-instant wake-ups.

## Prerequisites

- vLLM server must be started with `--enable-sleep-mode` and the `VLLM_SERVER_DEV_MODE=1` environment variable (required for sleep/wake endpoints).
- The vLLM server must be reachable at the URL provided to the wrapper.
- For sleep level 2: weights are discarded from GPU entirely on sleep. Wake requires reloading from disk. Ensure the model checkpoint is accessible.
- To enable automatic start‑if‑not‑running, provide a `--start-cmd` flag with a command that launches the vLLM server (e.g., a `docker run` command that includes `--enable-sleep-mode`). The wrapper will start the daemon if it is not reachable, then wait for it to become healthy.

## Installation

Build the binary from source:

```bash
go build -o vllm-wrapper ./cmd/vllm-wrapper
```

Or install via `go install`:

```bash
go install ./cmd/vllm-wrapper
```

## Usage in llama-swap

### As a model's `cmd`

Configure your model in `config.yaml` with a `cmd` that invokes `vllm-wrapper serve`:

```yaml
models:
  my-vllm-model:
    cmd: vllm-wrapper serve --vllm-url http://127.0.0.1:8000 --listen :${PORT} --start-cmd "docker run --rm -p 8000:8000 ... --enable-sleep-mode"
    # Optional flags:
    #   --sleep-level: sleep level to use when sleeping (default: 1)
    #   --sleep-mode: sleep mode for in-flight request handling (default: "wait")
    #     - wait: wait for in-flight requests to complete before sleeping
    #     - abort: immediately abort in-flight requests
    #   --health-path: health check path (default: /health)
    #   --wait-timeout: timeout waiting for daemon to become healthy (default: 120s)
```

When llama-swap starts the model, it will:
1. Check if the vLLM daemon is healthy by querying `${vllm-url}${health-path}` (default `/health`).
2. If healthy and awake, proceed to step 4.
3. If not healthy, attempt to wake the daemon by calling `${vllm-url}/wake_up`.
4. If wake‑up fails (e.g., connection refused), execute the `--start-cmd` to start the daemon.
5. Wait for the daemon to become healthy (polling the health path).
6. Start a reverse proxy from the port assigned by llama-swap (via `${PORT}`) to `${vllm-url}`.
7. Stay in the foreground as a proxy, allowing llama-swap to consider the model as running.

### As a model's `cmdStop`

Configure your model's `cmdStop` to invoke `vllm-wrapper sleep`:

```yaml
models:
  my-vllm-model:
    cmdStop: vllm-wrapper sleep --vllm-url http://127.0.0.1:8000 --sleep-level 2 --sleep-mode wait
    # Optional flags:
    #   --sleep-level: sleep level to use (default: 1)
    #   --sleep-mode: sleep mode for in-flight request handling (default: "wait")
```

When llama-swap stops the model, it will:
1. Send a sleep request to the vLLM daemon (POST `/sleep?level=2&mode=wait`).
2. Exit with status 0, leaving the vLLM daemon running but asleep.

## Example Configuration

Here is a complete example using vLLM with sleep level 2, demonstrating cold start on first swap‑in and full VRAM recovery on swap-out:

```yaml
models:
  qwen-7b-chat:
    cmd: vllm-wrapper serve --vllm-url http://127.0.0.1:8000 --listen :${PORT} --sleep-level 2 --start-cmd "VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen2.5-7B-Instruct --enable-sleep-mode --port 8000"
    cmdStop: vllm-wrapper sleep --vllm-url http://127.0.0.1:8000 --sleep-level 2 --sleep-mode wait --pid ${PID}
    # You may also want to set a TTL to automatically unload after a period of inactivity:
    ttl: 3600   # unload after 1 hour of inactivity
```

## How it works

### serve subcommand

1. **Health check**: Sends a GET request to `${vllm-url}${health-path}` (default `/health`). If the response is HTTP 200, the daemon is considered healthy and awake, and we proceed to step 4.
2. **Wake up**: If the health check fails (non‑200 or connection error), wake the daemon. The wake sequence depends on the sleep level:

   **Level 1 wake** (default): Sends `POST /wake_up` to restore weights from CPU RAM. Fast (~0.1-0.8s).

   **Level 2 wake** (when `--sleep-level 2`): Multi-step sequence:
   1. `POST /wake_up?tags=weights` — allocate GPU memory for weights
   2. `POST /collective_rpc` with `{"method": "reload_weights"}` — reload weights from disk
   3. `POST /wake_up?tags=kv_cache` — allocate GPU memory for KV cache
   4. `POST /reset_prefix_cache` — reset prefix cache (warning logged if this fails)

   Steps 1-3 must succeed or the wrapper exits with a fatal error. Step 4 failure is non-fatal (warning only).

   After a level-2 wake, health is polled with the full `--wait-timeout` (default 120s) to allow time for weight reload.

3. **Start daemon**: If the wake‑up fails (indicating the daemon is not running), execute the command specified by `--start-cmd` (run via `sh -c`). The wrapper starts the command as a child process, then waits for the daemon to become healthy by polling the health path.
4. **Reverse proxy**: Once the daemon is healthy, start an HTTP server listening on `${PORT}` (or the address provided to `--listen`) that proxies all requests to the vLLM upstream URL. The proxy preserves streaming responses by setting `X-Accel-Buffering: no`.

### sleep subcommand

1. Sends a POST request to `${vllm-url}/sleep?level=N&mode=M` where `N` is the sleep level (default 1) and `M` is the sleep mode (default "wait").
2. Upon receiving a successful response (HTTP 200), exits with status 0.

## Signal Handling

The `serve` subcommand distinguishes between normal swap-out and hard shutdown via OS signals:

- **SIGTERM / SIGINT**: Graceful proxy shutdown. The vLLM daemon is left running (sleeping or awake). This is the normal path when llama-swap swaps models — `cmdStop` puts the daemon to sleep, then the proxy exits.
- **SIGQUIT**: Hard shutdown. If the wrapper started the vLLM daemon via `--start-cmd`, it sends SIGKILL to the daemon process before exiting. This is used during config reloads to prevent orphaned vLLM processes.

Note: If the vLLM daemon was already running (not started by `--start-cmd`), SIGQUIT only shuts down the proxy — it does not kill externally-managed daemons.

## Notes

- The wrapper uses standard library only (no external dependencies).
- It is designed to be simple and robust.
- For production use, ensure the vLLM daemon is properly managed (e.g., restarted if it crashes) outside of this wrapper.
- The wrapper does not handle TLS certificates; if your vLLM server uses HTTPS, provide the appropriate URL and ensure the system's root CAs are configured.

## Building

```bash
go build -o vllm-wrapper ./cmd/vllm-wrapper
```

## Running Tests

```bash
go test ./...
```
