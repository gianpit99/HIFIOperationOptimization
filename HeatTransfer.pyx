def nextTemp(float T, float Q, float Tyn1, float Typ1, float Txn1, float Txp1, float dt, float alpha, float dx, float dy, float k):
    return ((Typ1 - 2.0 * T + Tyn1)/(dy**2) + (Txp1 - 2.0 * T + Txn1)/(dx**2) + Q/k) * alpha * dt + T
