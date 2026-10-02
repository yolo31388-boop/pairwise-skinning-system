"""蒙皮系统 - 含5个bug"""
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

class SkinningSystem:
    def __init__(self):
        self.bones: dict[str, Bone] = {}

    def calculate_skinning(self, vertex: Vertex) -> tuple:
        # bug1: 只用权重最高的骨骼，权重不归一化
        if not vertex.weights:
            return vertex.position
        best_bone = max(vertex.weights, key=vertex.weights.get)
        bone = self.bones.get(best_bone)
        if bone:
            return (vertex.position[0] + bone.position[0],
                    vertex.position[1] + bone.position[1])
        return vertex.position

    def apply_bone_transform(self, bone_id: str, translation: tuple, rotation: float) -> None:
        # bug2: 变换顺序错，不计算父子
        bone = self.bones.get(bone_id)
        if bone:
            bone.position = (bone.position[0] + translation[0],
                           bone.position[1] + translation[1])
            bone.rotation = rotation

    def apply_morph_target(self, vertex: tuple, morphs: dict) -> tuple:
        # bug3: 只应用一个，权重不限制
        for name, (offset, weight) in morphs.items():
            return (vertex[0] + offset[0] * weight,
                    vertex[1] + offset[1] * weight)
        return vertex

    def apply_constraint(self, bone_id: str, target: tuple) -> None:
        # bug4: 只做位置约束，不限制角度
        bone = self.bones.get(bone_id)
        if bone:
            bone.position = target

    def optimize_skinning(self, vertex: Vertex, max_bones: int = 4) -> Vertex:
        # bug5: 随机删除骨骼影响
        return vertex
