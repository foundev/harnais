package main

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
)

// Simple OS sandboxing for the bash tool, modeled on Codex CLI:
// a deny-by-default Seatbelt profile with writable roots passed as -D
// params to a hardcoded /usr/bin/sandbox-exec (resolving the helper via
// PATH would let a compromised PATH swap in an unsandboxed impostor).
//
// Posture: writes are confined to the command workdir plus the standard
// temp/cache locations toolchains need to function; reads are allowed (the
// agent already sees file contents — exfiltration is handled by denying
// network); network is always denied under the sandbox. The agent cannot
// switch this off per-call: only the --no-sandbox process flag escapes,
// so prompt injection cannot break out by asking.

// sandboxExecPath is hardcoded, mirroring Codex.
const sandboxExecPath = "/usr/bin/sandbox-exec"

// sandboxSupported reports whether OS-enforced sandboxing is implemented.
// macOS Seatbelt only for now; anything else runs unsandboxed with a
// one-time warning (Linux bubblewrap/Landlock is the roadmap item).
func sandboxSupported() bool {
	return runtime.GOOS == "darwin"
}

var sandboxWarnOnce sync.Once

func warnSandboxUnsupported() {
	sandboxWarnOnce.Do(func() {
		fmt.Fprintln(os.Stderr, "harnais: no OS sandbox on "+runtime.GOOS+" — bash running unsandboxed")
	})
}

// sandboxRoots resolves the writable roots for one command: the workdir
// plus temp/cache locations compilers and runtimes need. Symlinks are
// resolved the way Codex normalizes roots, since Seatbelt matches real
// paths (/tmp on macOS is a symlink to /private/tmp).
func sandboxRoots(workdir string) []string {
	resolve := func(p string) string {
		if abs, err := filepath.Abs(p); err == nil {
			p = abs
		}
		if real, err := filepath.EvalSymlinks(p); err == nil {
			p = real
		}
		return p
	}
	roots := []string{resolve(workdir), resolve(os.TempDir())}
	if home, err := os.UserHomeDir(); err == nil && home != "" {
		roots = append(roots, resolve(filepath.Join(home, ".cache")))
		roots = append(roots, resolve(filepath.Join(home, "Library", "Caches")))
	}
	seen := map[string]bool{}
	out := roots[:0]
	for _, r := range roots {
		if r != "" && !seen[r] {
			seen[r] = true
			out = append(out, r)
		}
	}
	return out
}

// sandboxProfile returns a deny-by-default Seatbelt profile. Reads stay
// open; each root gets a write subpath rule plus an anchor deny (mirroring
// Codex) so a command cannot unlink/replace the root directory itself.
func sandboxProfile(roots []string) string {
	var sb strings.Builder
	sb.WriteString(`(version 1)
(deny default)
(allow process-exec)
(allow process-fork)
(allow signal (target same-sandbox))
(allow process-info* (target same-sandbox))
(allow sysctl-read)
(allow mach-lookup (global-name "com.apple.system.opendirectoryd.libinfo"))
(allow file-read* (subpath "/"))
(allow file-read* file-write* (literal "/dev/null") (literal "/dev/zero") (literal "/dev/stdin") (literal "/dev/stdout") (literal "/dev/stderr") (literal "/dev/tty") (literal "/dev/ptmx"))
(allow pseudo-tty)
`)
	for i := range roots {
		fmt.Fprintf(&sb, "(allow file-write*\n  (subpath (param \"WRITABLE_ROOT_%d\")))\n", i)
		fmt.Fprintf(&sb, "(deny file-write-unlink (require-all (literal (param \"WRITABLE_ROOT_%d\")) (vnode-type DIRECTORY)))\n", i)
	}
	return sb.String()
}

// sandboxArgv builds the sandbox-exec invocation for `sh -c command`.
// Roots travel as -D params so paths with spaces or quotes never touch the
// policy source.
func sandboxArgv(roots []string, profile, command string) []string {
	argv := []string{sandboxExecPath, "-p", profile}
	for i, root := range roots {
		argv = append(argv, "-D", fmt.Sprintf("WRITABLE_ROOT_%d=%s", i, root))
	}
	return append(argv, "--", "sh", "-c", command)
}
