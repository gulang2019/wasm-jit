.PHONY: build build-tests run-tests clean

build:
	cmake -B build
	make -C build -j

build-tests:
	make -C tests

run-tests: build build-tests
	./grade.sh

clean:
	rm -r build
	make -C tests clean

