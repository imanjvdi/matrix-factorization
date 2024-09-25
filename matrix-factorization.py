import numpy as np

# نمونه ماتریس تعاملات
R = np.array([
    [5, 3, 0, 1, 4],
    [4, 0, 0, 1, 0],
    [1, 1, 0, 5, 0],
    [0, 0, 5, 4, 0],
    [0, 3, 4, 0, 0]
])

# پارامترهای اولیه
num_users, num_items = R.shape
k = 2  # تعداد فاکتورهای پنهان
steps = 5000  # تعداد تکرار الگوریتم
alpha = 0.002  # نرخ یادگیری
beta = 0.02  # مقدار منظم‌سازی (Regularization)

# مقداردهی اولیه ماتریس U و V با مقادیر تصادفی
U = np.random.rand(num_users, k)
V = np.random.rand(num_items, k)


# تابع هزینه برای محاسبه خطا
def cost_function(R, U, V, beta):
    prediction = np.dot(U, V.T)
    error = 0
    for i in range(len(R)):
        for j in range(len(R[i])):
            if R[i][j] > 0:  # فقط برای مقادیر غیر صفر محاسبه خطا
                error += (R[i][j] - np.dot(U[i, :], V[j, :])) ** 2
                # اضافه کردن Regularization
                error += beta * (np.sum(U[i, :] ** 2) + np.sum(V[j, :] ** 2))
    return error


# الگوریتم فاکتورگیری ماتریس با Gradient Descent
for step in range(steps):
    for i in range(len(R)):
        for j in range(len(R[i])):
            if R[i][j] > 0:  # فقط برای مقادیر غیر صفر به‌روز رسانی
                error_ij = R[i][j] - np.dot(U[i, :], V[j, :])

                # به‌روزرسانی مقادیر U و V
                for k_ in range(k):
                    U[i][k_] += alpha * (2 * error_ij * V[j][k_] - beta * U[i][k_])
                    V[j][k_] += alpha * (2 * error_ij * U[i][k_] - beta * V[j][k_])

    # محاسبه خطا و نمایش آن
    if step % 1000 == 0:
        error = cost_function(R, U, V, beta)
        print(f"Iteration: {step}, Error: {error}")

# نتیجه نهایی
predicted_matrix = np.dot(U, V.T)
print("\nماتریس اصلی:")
print(R)
print("\nماتریس پیش‌بینی‌شده:")
print(np.round(predicted_matrix, 2))
