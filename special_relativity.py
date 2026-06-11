import torch
import torch.nn.functional as F
from typing import Union

# ============================================================
# CONSTANTS
# ============================================================

C = 1.0  # скорость света в натуральных единицах

# ============================================================
# FOUR-VECTOR
# ============================================================

class FourVector:
    """
    4-вектор в пространстве-времени Минковского.
    Поддерживает как одиночные векторы (shape: 4,)
    так и батчи (shape: N, 4) для GPU вычислений.
    """

    def __init__(self, t, x, y, z):
        components = [t, x, y, z]
        tensors = [
            c if isinstance(c, torch.Tensor)
            else torch.tensor(c, dtype=torch.float64)
            for c in components
        ]
        self.data = torch.stack(tensors, dim=-1)

    @classmethod
    def from_tensor(cls, data: torch.Tensor) -> "FourVector":
        """Создать из тензора shape (4,) или (N, 4)"""
        obj = cls.__new__(cls)
        obj.data = data.to(torch.float64)
        return obj

    @classmethod
    def batch(cls, vectors: list) -> "FourVector":
        """Создать батч из списка FourVector"""
        return cls.from_tensor(torch.stack([v.data for v in vectors]))

    # ── арифметика ──────────────────────────────────────────

    def __add__(self, other: "FourVector") -> "FourVector":
        return FourVector.from_tensor(self.data + other.data)

    def __sub__(self, other: "FourVector") -> "FourVector":
        return FourVector.from_tensor(self.data - other.data)

    def __mul__(self, scalar) -> "FourVector":
        return FourVector.from_tensor(self.data * scalar)

    __rmul__ = __mul__

    def __neg__(self) -> "FourVector":
        return FourVector.from_tensor(-self.data)

    # ── компоненты ───────────────────────────────────────────

    @property
    def t(self) -> torch.Tensor:
        return self.data[..., 0]

    @property
    def x(self) -> torch.Tensor:
        return self.data[..., 1]

    @property
    def y(self) -> torch.Tensor:
        return self.data[..., 2]

    @property
    def z(self) -> torch.Tensor:
        return self.data[..., 3]

    @property
    def spatial(self) -> torch.Tensor:
        """3-вектор пространственных компонент"""
        return self.data[..., 1:]

    def __repr__(self) -> str:
        if self.data.dim() == 1:
            t, x, y, z = self.data.tolist()
            return f"FourVector(t={t:.4f}, x={x:.4f}, y={y:.4f}, z={z:.4f})"
        return f"FourVector(batch={self.data.shape[0]})"


# ============================================================
# MINKOWSKI METRIC  η = diag(+1, -1, -1, -1)
# ============================================================

class MinkowskiMetric:
    """
    Метрика Минковского со знаковым соглашением (+,-,-,-).
    """

    ETA = torch.diag(torch.tensor([1.0, -1.0, -1.0, -1.0],
                                   dtype=torch.float64))

    def inner(self, u: FourVector, v: FourVector) -> torch.Tensor:
        """Скалярное произведение: η_μν u^μ v^ν"""
        if u.data.dim() == 1 and v.data.dim() == 1:
            return u.t * v.t - torch.dot(u.spatial, v.spatial)
        # батч версия
        return (u.data * v.data * torch.tensor(
            [1, -1, -1, -1], dtype=torch.float64
        )).sum(dim=-1)

    def norm_squared(self, u: FourVector) -> torch.Tensor:
        """η_μν u^μ u^ν — может быть отрицательным"""
        return self.inner(u, u)

    def norm(self, u: FourVector) -> torch.Tensor:
        """|η_μν u^μ u^ν|^(1/2)"""
        return torch.sqrt(torch.abs(self.norm_squared(u)))

    def lower(self, u: FourVector) -> FourVector:
        """Опустить индекс: u^μ → u_μ"""
        sign = torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=torch.float64)
        return FourVector.from_tensor(u.data * sign)

    def raise_index(self, u: FourVector) -> FourVector:
        """Поднять индекс: для Минковского совпадает с lower"""
        return self.lower(u)

    def is_timelike(self, u: FourVector) -> torch.Tensor:
        return self.norm_squared(u) > 0

    def is_spacelike(self, u: FourVector) -> torch.Tensor:
        return self.norm_squared(u) < 0

    def is_lightlike(self, u: FourVector, tol=1e-10) -> torch.Tensor:
        return torch.abs(self.norm_squared(u)) < tol


# ============================================================
# LORENTZ BOOST
# ============================================================

class LorentzBoost:
    """
    Буст Лоренца вдоль произвольного направления.
    beta_vec: 3-вектор скорости в единицах c.
    """

    def __init__(self, beta_vec: Union[list, torch.Tensor]):
        beta_vec = torch.tensor(beta_vec, dtype=torch.float64)
        beta = torch.norm(beta_vec)

        if beta >= 1.0:
            raise ValueError(f"|β| = {beta:.4f} >= 1, скорость должна быть < c")

        self.beta_vec = beta_vec
        self.beta = beta
        self.gamma = 1.0 / torch.sqrt(1.0 - beta**2)
        self.L = self._build_matrix()

    def _build_matrix(self) -> torch.Tensor:
        γ = self.gamma
        β = self.beta
        n = self.beta_vec / β  # единичный вектор

        L = torch.zeros(4, 4, dtype=torch.float64)

        # временная строка/столбец
        L[0, 0] = γ
        L[0, 1:] = -γ * self.beta_vec
        L[1:, 0] = -γ * self.beta_vec

        # пространственный блок: δ_ij + (γ-1) n_i n_j
        L[1:, 1:] = torch.eye(3, dtype=torch.float64) + \
                    (γ - 1) * torch.outer(n, n)
        return L

    def __matmul__(self, v: FourVector) -> FourVector:
        if v.data.dim() == 1:
            return FourVector.from_tensor(self.L @ v.data)
        # батч: (N, 4) @ (4, 4).T
        return FourVector.from_tensor(v.data @ self.L.T)

    def inverse(self) -> "LorentzBoost":
        """Обратный буст — просто меняем знак скорости"""
        return LorentzBoost(-self.beta_vec)

    def __repr__(self) -> str:
        return (f"LorentzBoost(β={self.beta_vec.tolist()}, "
                f"γ={self.gamma:.4f})")


# ============================================================
# PHYSICAL QUANTITIES
# ============================================================

def gamma_factor(beta: torch.Tensor) -> torch.Tensor:
    """Фактор Лоренца γ = 1 / √(1 - β²)"""
    return 1.0 / torch.sqrt(1.0 - beta**2)


def four_velocity(v_spatial: Union[list, torch.Tensor]) -> FourVector:
    """
    4-скорость из обычной 3-скорости.
    U^μ = γ(c, v) → в натуральных единицах γ(1, v)
    """
    v = torch.tensor(v_spatial, dtype=torch.float64)
    beta = torch.norm(v)
    γ = gamma_factor(beta)
    return FourVector(γ, *(γ * v))


def four_momentum(mass: float,
                  v_spatial: Union[list, torch.Tensor]) -> FourVector:
    """
    4-импульс: p^μ = m U^μ = (E/c, p)
    """
    U = four_velocity(v_spatial)
    return mass * U


def kinetic_energy(mass: float,
                   v_spatial: Union[list, torch.Tensor]) -> torch.Tensor:
    """Кинетическая энергия: T = (γ - 1) m c²"""
    v = torch.tensor(v_spatial, dtype=torch.float64)
    beta = torch.norm(v)
    γ = gamma_factor(beta)
    return (γ - 1) * mass


def invariant_mass(p: FourVector,
                   metric: MinkowskiMetric) -> torch.Tensor:
    """
    Инвариантная масса: m² = η_μν p^μ p^ν
    """
    return torch.sqrt(torch.abs(metric.norm_squared(p)))


def spacetime_interval(u: FourVector, v: FourVector,
                       metric: MinkowskiMetric) -> torch.Tensor:
    """Интервал между двумя событиями"""
    return metric.norm_squared(u - v)


def doppler_factor(beta: float, cos_theta: float) -> float:
    """
    Релятивистский эффект Доплера.
    cos_theta: косинус угла между направлением движения и наблюдателем
    """
    γ = 1.0 / (1.0 - beta**2) ** 0.5
    return 1.0 / (γ * (1.0 - beta * cos_theta))


def aberration(theta: torch.Tensor, beta: float) -> torch.Tensor:
    """
    Аберрация света: угол в движущейся системе отсчёта.
    theta: угол в покоящейся системе
    """
    γ = gamma_factor(torch.tensor(beta))
    cos_t = torch.cos(theta)
    cos_prime = (cos_t - beta) / (1.0 - beta * cos_t)
    return torch.arccos(cos_prime)


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":
    η = MinkowskiMetric()

    # 4-вектор положения
    event = FourVector(5.0, 3.0, 0.0, 0.0)
    print(event)
    print("norm²:", η.norm_squared(event))

    # буст вдоль x с β = 0.6
    boost = LorentzBoost([0.6, 0.0, 0.0])
    print(boost)
    event_prime = boost @ event
    print("после буста:", event_prime)

    # инвариантность нормы
    print("норма до:", η.norm(event).item())
    print("норма после:", η.norm(event_prime).item())

    # 4-импульс фотона (m=0, движется вдоль x)
    p_photon = FourVector(1.0, 1.0, 0.0, 0.0)
    print("фотон lightlike:", η.is_lightlike(p_photon).item())

    # батч — 1000 бустов на GPU если есть
    device = "cuda" if torch.cuda.is_available() else "cpu"
    N = 1000
    data = torch.randn(N, 4, dtype=torch.float64).to(device)
    data[:, 0] = torch.abs(data[:, 0]) + 2.0  # временная компонента > 0
    batch = FourVector.from_tensor(data)
    boost.L = boost.L.to(device)
    result = boost @ batch
    print(f"батч буст {N} векторов на {device}: {result.data.shape}")