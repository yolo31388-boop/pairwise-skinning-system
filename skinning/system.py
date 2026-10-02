"""蒙皮系统

修复点:
1. calculate_skinning: 多骨骼权重混合 + 权重归一化
2. apply_bone_transform: 缩放 -> 旋转 -> 平移 的顺序, 并传播父子层级
3. apply_morph_target: 多变形目标叠加, 权重钳制在 [0, 1]
4. apply_constraint: 位置 + 旋转约束, 限制关节角度 (防反关节)
5. optimize_skinning: 保留权重最高的骨骼并重新归一化, 支持 GPU 蒙皮缓冲
"""
import math
from dataclasses import dataclass, field


def _clamp(value, low, high):
    return max(low, min(high, value))


def _rotate(point, degrees):
    rad = math.radians(degrees)
    c, s = math.cos(rad), math.sin(rad)
    x, y = point
    return (x * c - y * s, x * s + y * c)


@dataclass
class Bone:
    bid: str
    position: tuple
    rotation: float  # 角度制, 世界空间
    parent: str = None
    scale: float = 1.0


@dataclass
class Vertex:
    position: tuple
    weights: dict  # bone_id -> weight


@dataclass
class GPUSkinningBuffer:
    """上传到 GPU 的扁平蒙皮缓冲。

    positions/bone_indices/bone_weights 逐顶点对齐, bone_matrices 为每根
    骨骼的 3x3 仿射矩阵 (2D 齐次坐标), GPU vertex shader 中做一次矩阵混合。
    """
    positions: list
    bone_indices: list
    bone_weights: list
    bone_matrices: list
    max_bones: int = 4

    def skin(self):
        results = []
        for position, indices, weights in zip(self.positions,
                                              self.bone_indices,
                                              self.bone_weights):
            x = y = 0.0
            for idx, weight in zip(indices, weights):
                if weight <= 0.0:
                    continue
                matrix = self.bone_matrices[idx]
                px = matrix[0][0] * position[0] + matrix[0][1] * position[1] + matrix[0][2]
                py = matrix[1][0] * position[0] + matrix[1][1] * position[1] + matrix[1][2]
                x += weight * px
                y += weight * py
            results.append((x, y))
        return results


class SkinningSystem:
    def __init__(self):
        self.bones: dict[str, Bone] = {}

    @staticmethod
    def compose_trs(translation, rotation, scale, point):
        """按 缩放 -> 旋转 -> 平移 的顺序对一个局部点做变换。"""
        point = (point[0] * scale, point[1] * scale)
        point = _rotate(point, rotation)
        return (point[0] + translation[0], point[1] + translation[1])

    def _bone_matrix(self, bone):
        """骨骼世界矩阵: 平移 * 旋转 * 缩放 (作用于顶点时先缩放)。"""
        rad = math.radians(bone.rotation)
        c, s = math.cos(rad), math.sin(rad)
        sx = sy = bone.scale
        tx, ty = bone.position
        return [
            [c * sx, -s * sy, tx],
            [s * sx,  c * sy, ty],
            [0.0,     0.0,    1.0],
        ]

    def calculate_skinning(self, vertex: Vertex) -> tuple:
        """多骨骼线性混合蒙皮 (LBS), 权重按有效骨骼重新归一化。"""
        valid = {bid: weight for bid, weight in vertex.weights.items()
                 if bid in self.bones and weight > 0}
        total = sum(valid.values())
        if total <= 0:
            return tuple(vertex.position)

        x = y = 0.0
        for bid, weight in valid.items():
            normalized = weight / total
            bone = self.bones[bid]
            transformed = self.compose_trs(
                bone.position, bone.rotation, bone.scale, vertex.position)
            x += normalized * transformed[0]
            y += normalized * transformed[1]
        return (x, y)

    def _descendants(self, bone_id):
        result = []
        frontier = [bone_id]
        while frontier:
            current = frontier.pop()
            for bone in self.bones.values():
                if bone.parent == current:
                    result.append(bone)
                    frontier.append(bone.bid)
        return result

    def apply_bone_transform(self, bone_id: str, translation: tuple,
                             rotation: float, scale: float = 1.0) -> None:
        """应用骨骼局部变换, 顺序为 缩放 -> 旋转 -> 平移, 并联动所有子骨骼。"""
        bone = self.bones.get(bone_id)
        if bone is None:
            return

        old_origin = bone.position
        old_rotation = bone.rotation
        old_scale = bone.scale

        parent = self.bones.get(bone.parent) if bone.parent else None
        parent_origin = parent.position if parent else (0.0, 0.0)
        parent_rotation = parent.rotation if parent else 0.0
        parent_scale = parent.scale if parent else 1.0

        # 先把骨骼当前位置换算到父骨骼局部空间
        local = ((old_origin[0] - parent_origin[0]) / parent_scale,
                 (old_origin[1] - parent_origin[1]) / parent_scale)
        local = _rotate(local, -parent_rotation)
        # 局部空间内: 缩放 -> 旋转 -> 平移
        local = self.compose_trs(translation, rotation, scale, local)
        # 再映射回世界空间
        local = _rotate(local, parent_rotation)
        new_origin = (parent_origin[0] + parent_scale * local[0],
                      parent_origin[1] + parent_scale * local[1])

        new_rotation = parent_rotation + rotation
        new_scale = parent_scale * scale

        bone.position = new_origin
        bone.rotation = new_rotation
        bone.scale = new_scale

        # 子骨骼随父骨骼平移、绕父骨骼旋转、按比例缩放
        delta_rotation = new_rotation - old_rotation
        scale_ratio = new_scale / old_scale if old_scale else 1.0
        for child in self._descendants(bone_id):
            relative = ((child.position[0] - old_origin[0]),
                        (child.position[1] - old_origin[1]))
            relative = (relative[0] * scale_ratio,
                        relative[1] * scale_ratio)
            relative = _rotate(relative, delta_rotation)
            child.position = (new_origin[0] + relative[0],
                              new_origin[1] + relative[1])
            child.rotation += delta_rotation
            child.scale *= scale_ratio

    def apply_morph_target(self, vertex: tuple, morphs: dict) -> tuple:
        """叠加全部变形目标, 每个权重钳制到 [0, 1], 防止模型爆炸。"""
        x, y = vertex
        for offset, weight in morphs.values():
            clamped = _clamp(weight, 0.0, 1.0)
            x += offset[0] * clamped
            y += offset[1] * clamped
        return (x, y)

    def apply_constraint(self, bone_id: str, target: tuple,
                         max_angle: float = 180.0,
                         min_angle: float = -180.0,
                         constrain_position: bool = True,
                         constrain_rotation: bool = True) -> None:
        """位置约束 + 旋转约束, 旋转角被限制在 [min_angle, max_angle]。"""
        bone = self.bones.get(bone_id)
        if bone is None:
            return

        if constrain_position:
            bone.position = (target[0], target[1])

        if constrain_rotation:
            parent = self.bones.get(bone.parent) if bone.parent else None
            origin = parent.position if parent else (0.0, 0.0)
            dx = target[0] - origin[0]
            dy = target[1] - origin[1]
            if dx != 0.0 or dy != 0.0:
                desired = math.degrees(math.atan2(dy, dx))
                bone.rotation = _clamp(desired, min_angle, max_angle)

    def optimize_skinning(self, vertex: Vertex, max_bones: int = 4) -> Vertex:
        """保留权重最高的 max_bones 根骨骼, 其余剔除并重新归一化。"""
        items = list(vertex.weights.items())
        if len(items) > max_bones:
            items = sorted(items, key=lambda item: item[1], reverse=True)[:max_bones]
        total = sum(weight for _, weight in items)
        if total > 0:
            weights = {bid: weight / total for bid, weight in items}
        else:
            weights = dict(items)
        return Vertex(vertex.position, weights)

    def build_gpu_buffer(self, vertices, max_bones: int = 4) -> GPUSkinningBuffer:
        """把优化后的顶点权重和骨骼矩阵打包成 GPU 蒙皮缓冲。"""
        bone_ids = list(self.bones.keys())
        bone_index = {bid: i for i, bid in enumerate(bone_ids)}
        matrices = [self._bone_matrix(self.bones[bid]) for bid in bone_ids]

        positions = []
        all_indices = []
        all_weights = []
        for vertex in vertices:
            optimized = self.optimize_skinning(vertex, max_bones)
            positions.append(tuple(vertex.position))
            indices = []
            weights = []
            for bid, weight in optimized.weights.items():
                indices.append(bone_index[bid])
                weights.append(weight)
            while len(indices) < max_bones:
                indices.append(0)
                weights.append(0.0)
            all_indices.append(indices)
            all_weights.append(weights)

        return GPUSkinningBuffer(
            positions=positions,
            bone_indices=all_indices,
            bone_weights=all_weights,
            bone_matrices=matrices,
            max_bones=max_bones,
        )

    def gpu_skinning(self, vertices, max_bones: int = 4) -> list:
        """GPU 蒙皮路径: 每帧只上传矩阵, 混合在着色器/扁平缓冲中完成。"""
        return self.build_gpu_buffer(vertices, max_bones).skin()
