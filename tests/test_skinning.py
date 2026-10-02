"""蒙皮系统 - 修复验证测试"""
import math
import pytest, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from skinning.system import SkinningSystem, Bone, Vertex, GPUSkinningBuffer


def make_system():
    ss = SkinningSystem()
    ss.bones["b1"] = Bone("b1", (1, 0), 0)
    ss.bones["b2"] = Bone("b2", (0, 1), 0)
    return ss


class TestMultiBone:
    def test_multiple_bones_contribute(self):
        ss = make_system()
        v = Vertex((0, 0), {"b1": 0.5, "b2": 0.5})
        result = ss.calculate_skinning(v)
        assert result[0] > 0 and result[1] > 0

    def test_weights_are_normalized(self):
        ss = make_system()
        v = Vertex((0, 0), {"b1": 2.0, "b2": 2.0})
        result = ss.calculate_skinning(v)
        assert result == pytest.approx((0.5, 0.5))

    def test_weighted_blend_proportions(self):
        ss = make_system()
        v = Vertex((0, 0), {"b1": 0.75, "b2": 0.25})
        result = ss.calculate_skinning(v)
        assert result == pytest.approx((0.75, 0.25))

    def test_unknown_bone_ignored_and_renormalized(self):
        ss = make_system()
        v = Vertex((0, 0), {"b1": 0.5, "ghost": 0.5})
        result = ss.calculate_skinning(v)
        assert result == pytest.approx((1.0, 0.0))


class TestBoneHierarchy:
    def test_child_follows_parent(self):
        ss = SkinningSystem()
        ss.bones["parent"] = Bone("parent", (0, 0), 0)
        ss.bones["child"] = Bone("child", (1, 0), 0, parent="parent")
        ss.apply_bone_transform("parent", (5, 0), 0)
        assert ss.bones["child"].position[0] > 1

    def test_transform_order_scale_rotate_translate(self):
        ss = SkinningSystem()
        ss.bones["root"] = Bone("root", (0, 0), 0)
        ss.bones["arm"] = Bone("arm", (1, 0), 0, parent="root")
        ss.apply_bone_transform("arm", (1, 0), 90, scale=2.0)
        # 局部 (1,0) -> 缩放 (2,0) -> 旋转90 (0,2) -> 平移 (1,2)
        assert ss.bones["arm"].position == pytest.approx((1.0, 2.0))
        assert ss.bones["arm"].rotation == pytest.approx(90.0)
        assert ss.bones["arm"].scale == pytest.approx(2.0)

    def test_child_rotates_with_parent(self):
        ss = SkinningSystem()
        ss.bones["parent"] = Bone("parent", (0, 0), 0)
        ss.bones["child"] = Bone("child", (1, 0), 0, parent="parent")
        ss.apply_bone_transform("parent", (0, 0), 90)
        assert ss.bones["child"].position == pytest.approx((0.0, 1.0))
        assert ss.bones["child"].rotation == pytest.approx(90.0)

    def test_grandchild_follows_chain(self):
        ss = SkinningSystem()
        ss.bones["a"] = Bone("a", (0, 0), 0)
        ss.bones["b"] = Bone("b", (1, 0), 0, parent="a")
        ss.bones["c"] = Bone("c", (2, 0), 0, parent="b")
        ss.apply_bone_transform("a", (10, 0), 0)
        assert ss.bones["b"].position[0] == pytest.approx(11.0)
        assert ss.bones["c"].position[0] == pytest.approx(12.0)


class TestMorph叠加:
    def test_multiple_morphs_apply(self):
        ss = SkinningSystem()
        morphs = {"m1": ((1, 0), 0.5), "m2": ((0, 1), 0.5)}
        result = ss.apply_morph_target((0, 0), morphs)
        assert result[0] > 0 and result[1] > 0

    def test_morph_weight_clamped(self):
        ss = SkinningSystem()
        morphs = {"m1": ((1, 0), 5.0)}
        result = ss.apply_morph_target((0, 0), morphs)
        assert result[0] <= 1.0

    def test_morph_negative_weight_clamped(self):
        ss = SkinningSystem()
        morphs = {"m1": ((1, 0), -2.0)}
        result = ss.apply_morph_target((0, 0), morphs)
        assert result == (0, 0)


class TestRotationConstraint:
    def test_constraint_limits_angle(self):
        ss = SkinningSystem()
        ss.bones["b1"] = Bone("b1", (0, 0), 0)
        ss.apply_constraint("b1", (1, 0))
        assert ss.bones["b1"].rotation <= 180

    def test_rotation_constraint_tracks_target(self):
        ss = SkinningSystem()
        ss.bones["b1"] = Bone("b1", (0, 0), 0)
        ss.apply_constraint("b1", (0, 1))
        assert ss.bones["b1"].rotation == pytest.approx(90.0)

    def test_anti_hyperextension_clamp(self):
        ss = SkinningSystem()
        ss.bones["elbow"] = Bone("elbow", (1, 0), 0, parent=None)
        ss.apply_constraint("elbow", (-1, 0), min_angle=0.0, max_angle=150.0)
        assert 0.0 <= ss.bones["elbow"].rotation <= 150.0


class TestOptimize:
    def test_optimize_keeps_important_bones(self):
        ss = SkinningSystem()
        v = Vertex((0, 0), {"b1": 0.8, "b2": 0.1, "b3": 0.1})
        result = ss.optimize_skinning(v, max_bones=2)
        assert "b1" in result.weights

    def test_optimize_renormalizes(self):
        ss = SkinningSystem()
        v = Vertex((0, 0), {"b1": 0.8, "b2": 0.1, "b3": 0.1})
        result = ss.optimize_skinning(v, max_bones=2)
        assert sum(result.weights.values()) == pytest.approx(1.0)
        assert len(result.weights) == 2

    def test_gpu_skinning_matches_cpu(self):
        ss = make_system()
        vertices = [Vertex((0, 0), {"b1": 0.5, "b2": 0.5}),
                    Vertex((1, 1), {"b1": 0.25, "b2": 0.75})]
        cpu = [ss.calculate_skinning(v) for v in vertices]
        gpu = ss.gpu_skinning(vertices)
        for c, g in zip(cpu, gpu):
            assert g == pytest.approx(c)

    def test_gpu_buffer_structure(self):
        ss = make_system()
        buffer = ss.build_gpu_buffer([Vertex((0, 0), {"b1": 1.0})], max_bones=4)
        assert isinstance(buffer, GPUSkinningBuffer)
        assert len(buffer.positions) == 1
        assert len(buffer.bone_indices[0]) == 4
        assert len(buffer.bone_matrices) == 2
