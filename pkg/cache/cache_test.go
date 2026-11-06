package cache

import (
	"path/filepath"
	"testing"
)

func TestSetGetRoundTrip(t *testing.T) {
	c, err := Open(filepath.Join(t.TempDir(), "c.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	key := Key("judge-v1", "q", "a", "ref", 0)
	if _, ok, err := c.Get(key); err != nil || ok {
		t.Fatalf("expected a miss before any Set, ok=%v err=%v", ok, err)
	}

	want := Entry{Score: 0.8, Pass: true, Rationale: "good"}
	if err := c.Set(key, want); err != nil {
		t.Fatal(err)
	}
	got, ok, err := c.Get(key)
	if err != nil || !ok {
		t.Fatalf("expected a hit after Set, ok=%v err=%v", ok, err)
	}
	if got != want {
		t.Fatalf("expected %+v, got %+v", want, got)
	}
}

func TestKeyChangesWithAnyInput(t *testing.T) {
	base := Key("j", "q", "a", "ref", 0)
	variants := []string{
		Key("j2", "q", "a", "ref", 0),
		Key("j", "q2", "a", "ref", 0),
		Key("j", "q", "a2", "ref", 0),
		Key("j", "q", "a", "ref2", 0),
		Key("j", "q", "a", "ref", 1),
	}
	for i, v := range variants {
		if v == base {
			t.Fatalf("variant %d produced the same key as the base — cache key is not sensitive to that input", i)
		}
	}
}

func TestSetOverwritesExistingEntry(t *testing.T) {
	c, err := Open(filepath.Join(t.TempDir(), "c.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	key := Key("j", "q", "a", "ref", 0)
	c.Set(key, Entry{Score: 0.1})
	c.Set(key, Entry{Score: 0.9})
	got, ok, err := c.Get(key)
	if err != nil || !ok || got.Score != 0.9 {
		t.Fatalf("expected the second Set to overwrite, got %+v ok=%v err=%v", got, ok, err)
	}
}
