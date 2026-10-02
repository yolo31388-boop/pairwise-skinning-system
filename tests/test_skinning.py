"""蒙皮系统 - 红态测试"""
import pytest, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from skinning.system import SkinningSystem, Bone, Vertex

class TestMultiBone:
    def test_multiple_bones_contribute(self):
        ss = SkinningSystem()
        ss.bones["b1"] = Bone("b1", (1,0), 0)
        ss.bones["b2"] = Bone("b2", (0,1), 0)
        v = Vertex((0,0), {"b1": 0.5, "b2": 0.5})
        result = ss.calculate_skinning(v)
        assert result[0] > 0 and result[1] > 0

class TestBoneHierarchy:
    def test_child_follows_parent(self):
        ss = SkinningSystem()
        ss.bones["parent"] = Bone("parent", (0,0), 0)
        ss.bones["child"] = Bone("child", (1,0), 0, parent="parent")
        ss.apply_bone_transform("parent", (5,0), 0)
        assert ss.bones["child"].position[0] > 1

class TestMorph叠加:
    def test_multiple_morphs_apply(self):
        ss = SkinningSystem()
        morphs = {"m1": ((1,0), 0.5), "m2": ((0,1), 0.5)}
        result = ss.apply_morph_target((0,0), morphs)
        assert result[0] > 0 and result[1] > 0

class TestRotationConstraint:
    def test_constraint_limits_angle(self):
        ss = SkinningSystem()
        ss.bones["b1"] = Bone("b1", (0,0), 0)
        ss.apply_constraint("b1", (1,0))
        assert ss.bones["b1"].rotation <= 180

class TestOptimize:
    def test_optimize_keeps_important_bones(self):
        ss = SkinningSystem()
        v = Vertex((0,0), {"b1": 0.8, "b2": 0.1, "b3": 0.1})
        result = ss.optimize_skinning(v, max_bones=2)
        assert "b1" in result.weights
