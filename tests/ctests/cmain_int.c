__attribute__((export_name("main")))
int main_stub(int d0) {
  for (int i = 1; i < 7; i++) {
    d0 = (d0 / i ) - 13;
  }
  return d0;
}
