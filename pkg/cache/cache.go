// Package cache gives judge calls a content-addressable, persistent cache:
// re-running an unchanged eval (same question, same answer, same
// reference, same judge, same sample index) should never re-pay for a
// judge call that already happened. The sample index is deliberately part
// of the key — see pkg/eval's doc comment for why that's what lets a
// single run sample a noisy judge K independent times while still making a
// REPEAT of that exact run free.
package cache

import (
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"fmt"

	_ "modernc.org/sqlite"
)

type Entry struct {
	Score     float64
	Pass      bool
	Rationale string
}

type Cache struct {
	db *sql.DB
}

func Open(path string) (*Cache, error) {
	db, err := sql.Open("sqlite", "file:"+path+"?_pragma=journal_mode(WAL)&_pragma=busy_timeout(5000)")
	if err != nil {
		return nil, err
	}
	db.SetMaxOpenConns(1)
	if _, err := db.Exec(`CREATE TABLE IF NOT EXISTS judge_cache (key TEXT PRIMARY KEY, value TEXT NOT NULL)`); err != nil {
		db.Close()
		return nil, fmt.Errorf("migrate: %w", err)
	}
	return &Cache{db: db}, nil
}

func (c *Cache) Close() error { return c.db.Close() }

// Key derives a stable cache key from everything that determines a judge
// call's outcome. Changing the judge's model/prompt version should be
// reflected in judgeID by the caller (e.g. "gpt-4o-mini@rubric-v2") so a
// prompt change correctly invalidates old cached scores instead of
// silently reusing stale ones.
func Key(judgeID, question, answer, reference string, sampleIndex int) string {
	h := sha256.New()
	fmt.Fprintf(h, "%s\x00%s\x00%s\x00%s\x00%d", judgeID, question, answer, reference, sampleIndex)
	return hex.EncodeToString(h.Sum(nil))
}

func (c *Cache) Get(key string) (Entry, bool, error) {
	var raw string
	err := c.db.QueryRow(`SELECT value FROM judge_cache WHERE key = ?`, key).Scan(&raw)
	if err == sql.ErrNoRows {
		return Entry{}, false, nil
	}
	if err != nil {
		return Entry{}, false, err
	}
	var e Entry
	if err := json.Unmarshal([]byte(raw), &e); err != nil {
		return Entry{}, false, err
	}
	return e, true, nil
}

func (c *Cache) Set(key string, e Entry) error {
	raw, err := json.Marshal(e)
	if err != nil {
		return err
	}
	_, err = c.db.Exec(`INSERT INTO judge_cache (key, value) VALUES (?, ?)
		ON CONFLICT(key) DO UPDATE SET value = excluded.value`, key, string(raw))
	return err
}
