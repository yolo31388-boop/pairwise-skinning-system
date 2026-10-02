"""蒙皮系统"""
import math
from dataclasses import dataclass, field

@dataclass
class Bone:
    bid: str
    position: tuple
    rotation: float
    parent: str = None

@dataclass
class Vertex:
    position: tuple
    weights: dict  # bone_id -> weight

def _normalize_weights(weights: dict) -> dict:
    """权重归一化，保证所有权重之和为1"""
    total = sum(weights.values())
    if total <= 0:
        return dict(weights)
    return {k: v / total for k, v in weights.items()}

class SkinningSystem:
    def __init__(self):
        self.bones: dict[str, Bone] = {}
        self.use_gpu = self._gpu_available()

    @staticmethod
    def _gpu_available() -> bool:
        try:
            import numpy  # noqa: F401
            return True
        except ImportError:
            return False

    def calculate_skinning(self, vertex: Vertex) -> tuple:
        # fix1: 所有骨骼按权重加权混合，且权重归一化
        if not vertex.weights:
            return vertex.position
        weights = _normalize_weights(vertex.weights)
        x, y = vertex.position[0], vertex.position[1]
        for bone_id, w in weights.items():
            bone = self.bones.get(bone_id)
            if bone:
                x += bone.position[0] * w
                y += bone.position[1] * w
        return (x, y)

    def calculate_skinning_batch(self, vertices: list, use_gpu: bool = None) -> list:
        """批量蒙皮：有GPU(numpy)时走向量化路径，否则回退CPU"""
        if use_gpu is None:
            use_gpu = self.use_gpu
        if use_gpu and self._gpu_available():
            return self._skinning_gpu(vertices)
        return [self.calculate_skinning(v) for v in vertices]

    def _skinning_gpu(self, vertices: list) -> list:
        import numpy as np
        if not vertices:
            return []
        bone_ids = list(self.bones)
        bone_pos = np.array([self.bones[b].position[:2] for b in bone_ids], dtype=float)
        bone_index = {b: i for i, b in enumerate(bone_ids)}
        positions = np.array([v.position[:2] for v in vertices], dtype=float)
        wmat = np.zeros((len(vertices), len(bone_ids)))
        for row, v in enumerate(vertices):
            for bid, w in _normalize_weights(v.weights).items():
                if bid in bone_index:
                    wmat[row, bone_index[bid]] = w
        result = positions + wmat @ bone_pos
        return [tuple(row) for row in result]

    def apply_bone_transform(self, bone_id: str, translation: tuple, rotation: float,
                             scale: float = 1.0, _pivot: tuple = None) -> None:
        # fix2: 变换顺序为 缩放->旋转->平移(绕骨骼自身 pivot)，并递归带动子骨骼
        bone = self.bones.get(bone_id)
        if not bone:
            return
        pivot = _pivot if _pivot is not None else bone.position
        rad = math.radians(rotation)
        cos_r, sin_r = math.cos(rad), math.sin(rad)
        sx = (bone.position[0] - pivot[0]) * scale
        sy = (bone.position[1] - pivot[1]) * scale
        bone.position = (sx * cos_r - sy * sin_r + pivot[0] + translation[0],
                         sx * sin_r + sy * cos_r + pivot[1] + translation[1])
        if _pivot is None:
            bone.rotation = rotation
        else:
            bone.rotation += rotation
        for child_id, child in self.bones.items():
            if child.parent == bone_id:
                self.apply_bone_transform(child_id, translation, rotation, scale,
                                          _pivot=pivot)

    def apply_morph_target(self, vertex: tuple, morphs: dict) -> tuple:
        # fix3: 所有变形目标叠加，权重限制在[0,1]
        x, y = vertex[0], vertex[1]
        for name, (offset, weight) in morphs.items():
            w = max(0.0, min(1.0, weight))
            x += offset[0] * w
            y += offset[1] * w
        return (x, y)

    def apply_constraint(self, bone_id: str, target: tuple, max_angle: float = 180.0) -> None:
        # fix4: 位置+旋转约束，旋转角度限制在[-max_angle, max_angle]
        bone = self.bones.get(bone_id)
        if not bone:
            return
        dx = target[0] - bone.position[0]
        dy = target[1] - bone.position[1]
        angle = math.degrees(math.atan2(dy, dx))
        angle = max(-max_angle, min(max_angle, angle))
        bone.rotation = angle
        bone.position = target

    def optimize_skinning(self, vertex: Vertex, max_bones: int = 4) -> Vertex:
        # fix5: 保留权重最高的骨骼影响，剩余权重重新归一化
        if len(vertex.weights) <= max_bones:
            return Vertex(vertex.position, _normalize_weights(vertex.weights))
        top = sorted(vertex.weights.items(), key=lambda kv: kv[1], reverse=True)[:max_bones]
        return Vertex(vertex.position, _normalize_weights(dict(top)))
