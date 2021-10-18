from locker import locker
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

n_neighbours = 10

fn = locker(n_neighbors=n_neighbours)
x = np.linspace(-10, 10, 101)
y = x ** 2 + 20 * np.random.normal(size=x.shape)
# x = np.stack(
#     [
#         x,
#         x + 0.1 * np.random.normal(size=x.shape),
#     ],
#     axis=1,
# )
x = x.reshape(-1, 1)


fn.fit(x, y)
test_x = np.linspace(-20, 20, 201)
test_x = test_x.reshape(-1, 1)
# test_x = np.stack([test_x, test_x], axis=1)
test_y, test_y_unc = fn.predict(test_x)

plt.plot(test_x[:, 0], test_y, color='r', label='LoCKeR')
plt.fill_between(
    test_x[:, 0], test_y + test_y_unc, test_y - test_y_unc, color='r', alpha=0.5
)

neigh = KNeighborsRegressor(n_neighbors=n_neighbours)
neigh.fit(x, y)
plt.plot(test_x[:, 0], neigh.predict(test_x), color='g', label='KNN')


scaler_x, scaler_y, unc_scaler_y = (
    StandardScaler(),
    StandardScaler(),
    StandardScaler(with_std=False),
)
normalised_x, normalised_y = scaler_x.fit_transform(x), scaler_y.fit_transform(
    y.reshape(-1, 1)
)
unc_y_scaler = unc_scaler_y.fit(y.reshape(-1, 1))
gpr = GaussianProcessRegressor(
    kernel=kernels.RBF(length_scale=1, length_scale_bounds='fixed')
).fit(normalised_x, normalised_y.flatten())
gpr_pred, grp_unc = gpr.predict(scaler_x.transform(test_x), return_std=True)
plt.plot(
    test_x[:, 0],
    scaler_y.inverse_transform(gpr_pred),
    color='orange',
    label='GPR',
)
plt.fill_between(
    test_x[:, 0],
    scaler_y.inverse_transform(gpr_pred)
    + unc_y_scaler.inverse_transform(grp_unc),
    scaler_y.inverse_transform(gpr_pred)
    - unc_y_scaler.inverse_transform(grp_unc),
    color='orange',
    alpha=0.5,
)


plt.legend()
plt.ylim(-50, 400)
plt.scatter(x[:, 0], y)
plt.show()
