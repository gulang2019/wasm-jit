__attribute__((export_name("main")))
double main_stub(double d1, double d2) {
  for (int i = 0; i < 8; i++) {
    d1 = (d1 * 2.1) + d2;
  }
  return d1;
}
