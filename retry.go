package main

import (
	"context"
	"math"
	"math/rand"
	"time"
)

// Retry policy mirrors the Codex CLI request layer
// (codex-rs/codex-client/src/retry.rs with the provider defaults in
// codex-rs/model-provider-info/src/lib.rs): bounded attempts with
// exponential backoff and jitter. Transient 5xx and transport failures
// are retried; 429 is not retried by default and neither is anything
// else (401/4xx are auth or caller errors, handled elsewhere).
const defaultRequestMaxRetries = 4
const defaultRetryBaseDelay = 200 * time.Millisecond

type retryPolicy struct {
	maxAttempts    int
	baseDelay      time.Duration
	retry429       bool
	retry5xx       bool
	retryTransport bool
}

func defaultRetryPolicy() retryPolicy {
	return retryPolicy{
		maxAttempts:    defaultRequestMaxRetries,
		baseDelay:      defaultRetryBaseDelay,
		retry5xx:       true,
		retryTransport: true,
	}
}

// retryableStatus reports whether an HTTP status is worth another attempt.
func (p retryPolicy) retryableStatus(status int) bool {
	if status == 429 {
		return p.retry429
	}
	return p.retry5xx && status >= 500 && status <= 599
}

// delayFor returns the backoff before retry number attempt (1-based, like
// Codex's backoff(base, retry_attempt)): base, 2*base, 4*base, ... each
// with +/-10% jitter.
func (p retryPolicy) delayFor(attempt int) time.Duration {
	return retryBackoff(p.baseDelay, attempt)
}

func retryBackoff(base time.Duration, attempt int) time.Duration {
	if attempt <= 0 {
		return base
	}
	raw := int64(base)
	for i := 1; i < attempt; i++ {
		if raw > math.MaxInt64/2 {
			raw = math.MaxInt64
			break
		}
		raw *= 2
	}
	jitter := 0.9 + 0.2*rand.Float64()
	return time.Duration(float64(raw) * jitter)
}

// sleepRetry waits out a backoff but gives up early when the caller goes
// away, so retries never outlive cancellation.
func sleepRetry(ctx context.Context, d time.Duration) error {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}
