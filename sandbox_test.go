package main

import (
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestSandboxProfileDenyDefault(t *testing.T) {
	profile := sandboxProfile([]string{"/work"})
	if !strings.Contains(profile, "(deny default)") {
		t.Error("profile must be deny-by-default")
	}
	if !strings.Contains(profile, `(subpath (param "WRITABLE_ROOT_0"))`) {
		t.Error("profile must reference the writable root param")
	}
	if strings.Contains(profile, "network") {
		t.Error("profile must not allow network")
	}
	// Anchor rule: the root directory itself cannot be unlinked/replaced.
	if !strings.Contains(profile, "file-write-unlink") {
		t.Error("profile must anchor-deny unlinking the root")
	}
}

func TestSandboxArgv(t *testing.T) {
	argv := sandboxArgv([]string{"/w 1", "/tmp"}, "PROFILE", "echo hi")
	want := []string{
		"/usr/bin/sandbox-exec", "-p", "PROFILE",
		"-D", "WRITABLE_ROOT_0=/w 1",
		"-D", "WRITABLE_ROOT_1=/tmp",
		"--", "sh", "-c", "echo hi",
	}
	if len(argv) != len(want) {
		t.Fatalf("argv wrong: %q", argv)
	}
	for i := range want {
		if argv[i] != want[i] {
			t.Fatalf("argv[%d] = %q, want %q (full: %q)", i, argv[i], want[i], argv)
		}
	}
}

func TestSandboxRoots(t *testing.T) {
	dir := t.TempDir()
	roots := sandboxRoots(dir)
	if len(roots) < 2 {
		t.Fatalf("expected workdir plus temp/cache roots, got %q", roots)
	}
	resolved, _ := filepath.EvalSymlinks(dir)
	if roots[0] != resolved {
		t.Errorf("first root should be the resolved workdir: %q", roots)
	}
	for _, r := range roots {
		if !filepath.IsAbs(r) {
			t.Errorf("root must be absolute: %q", r)
		}
	}
}

func TestSandboxSupportedMatchesRuntime(t *testing.T) {
	if sandboxSupported() != (runtime.GOOS == "darwin") {
		t.Errorf("sandboxSupported() = %v on %s", sandboxSupported(), runtime.GOOS)
	}
}
