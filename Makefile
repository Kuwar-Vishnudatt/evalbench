.PHONY: build test test-race demo fmt vet clean

build:
	go build -o bin/evalbench ./cmd/evalbench

test:
	go test ./... -count=1

test-race:
	go test ./... -race -count=1

demo: build
	./scripts/run-demo.sh

fmt:
	gofmt -l .

vet:
	go vet ./...

clean:
	rm -rf bin/ *.db *.db-wal *.db-shm run.json
