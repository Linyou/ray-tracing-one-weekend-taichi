import taichi as ti
from vector import *
import ray
from material import Materials
import random
import numpy as np
from bvh import BVH
from taichi.math import vec3


@ti.func
def is_front_facing(ray_direction, normal):
    return ray_direction.dot(normal) < 0.0


@ti.func
def hit_sphere(center, radius, ray_origin, ray_direction, t_min, t_max):
    ''' Intersect a sphere of given radius and center and return
        if it hit and the least root. '''
    oc = ray_origin - center
    a = ray_direction.norm_sqr()
    half_b = oc.dot(ray_direction)
    c = (oc.norm_sqr() - radius**2)
    discriminant = (half_b**2) - a * c

    hit = discriminant >= 0.0
    root = -1.0
    if hit:
        sqrtd = discriminant**0.5
        root = (-half_b - sqrtd) / a

        if root < t_min or t_max < root:
            root = (-half_b + sqrtd) / a
            if root < t_min or t_max < root:
                hit = False

    return hit, root

@ti.func
def hit_cube(center, radius, ray_origin, ray_direction, t_min, t_max):
    ''' Intersect a cube of given radius and center and return
        if it hit and the least root. '''
    inv_d = 1. / ray_direction

    t_min_ = (center-radius-ray_origin)*inv_d
    t_max_ = (center+radius-ray_origin)*inv_d

    _t1 = ti.min(t_min_, t_max_)
    _t2 = ti.max(t_min_, t_max_)
    t1 = _t1.max()
    t2 = _t2.min()

    hit = (t2 > ti.max(t1, 0.))
    root = t1

    if root < t_min or t_max < root:
        hit = False

    return hit, root

@ti.func
def cube_normal(point, center):
    box_hit = point - center
    normal = box_hit / ti.abs(box_hit).max()
    for i in ti.static(range(3)):
        if ti.abs(normal[i]) >= 1.:
            normal[i] = ti.math.sign(normal[i])*1.0
        else:
            normal[i] = 0.
    return normal

class Sphere:
    def __init__(self, center, radius, material):
        self.center = center
        self.radius = radius
        self.material = material
        self.id = -1
        self.object_type = 0
        self.box_min = [
            self.center[0] - radius, self.center[1] - radius,
            self.center[2] - radius
        ]
        self.box_max = [
            self.center[0] + radius, self.center[1] + radius,
            self.center[2] + radius
        ]

    @property
    def bounding_box(self):
        return self.box_min, self.box_max


class Cube(Sphere):
    def __init__(self, center, radius, material):
        super().__init__(center, radius, material)
        self.object_type = 1


BRANCH = 1.0
LEAF = 0.0

SPHERE_TYPE = 0
CUBE_TYPE = 1


@ti.data_oriented
class World:
    def __init__(self):
        self.obj_list = []

    def add(self, sphere):
        sphere.id = len(self.obj_list)
        self.obj_list.append(sphere)

    def commit(self):
        ''' Commit should be called after all objects added.  
            Will compile bvh and materials. '''
        self.n = len(self.obj_list)

        self.materials = Materials(self.n)
        self.bvh = BVH(self.obj_list)
        self.radius = ti.field(ti.f32)
        self.center = ti.Vector.field(3, dtype=ti.f32)
        self.object_type = ti.field(ti.i32)
        ti.root.dense(ti.i, self.n).place(self.radius, self.center, self.object_type)

        self.bvh.build()

        for i in range(self.n):
            self.center[i] = self.obj_list[i].center
            self.radius[i] = self.obj_list[i].radius
            self.object_type[i] = self.obj_list[i].object_type
            self.materials.set(i, self.obj_list[i].material)

        del self.obj_list

    def bounding_box(self, i):
        return self.bvh_min(i), self.bvh_max(i)

    @ti.func
    def hit_all(self, ray_origin, ray_direction):
        ''' Intersects a ray against all objects. '''
        hit_anything = False
        t_min = 0.0001
        closest_so_far = 9999999999.9
        hit_index = 0
        p = Point(0.0, 0.0, 0.0)
        n = Vector(0.0, 0.0, 0.0)
        front_facing = True
        i = 0
        curr = self.bvh.bvh_root

        # walk the bvh tree
        while curr != -1:
            obj_id, left_id, right_id, next_id = self.bvh.get_full_id(curr)

            if obj_id != -1:
                # this is a leaf node, check the sphere
                hit, t = False, t_min

                if self.object_type[obj_id] == SPHERE_TYPE:
                    hit, t = hit_sphere(self.center[obj_id], self.radius[obj_id],
                                        ray_origin, ray_direction, t_min,
                                        closest_so_far)
                elif self.object_type[obj_id] == CUBE_TYPE:
                    hit, t = hit_cube(self.center[obj_id], self.radius[obj_id],
                                        ray_origin, ray_direction, t_min,
                                        closest_so_far)

                if hit:
                    hit_anything = True
                    closest_so_far = t
                    hit_index = obj_id
                curr = next_id
            else:
                if self.bvh.hit_aabb(curr, ray_origin, ray_direction, t_min,
                                     closest_so_far):
                    # add left and right children
                    if left_id != -1:
                        curr = left_id
                    elif right_id != -1:
                        curr = right_id
                    else:
                        curr = next_id
                else:
                    curr = next_id

        if hit_anything:
            p = ray.at(ray_origin, ray_direction, closest_so_far)
            # n = (p - self.center[hit_index]) / self.radius[hit_index]
            if self.object_type[hit_index] == SPHERE_TYPE:
                n = (p - self.center[hit_index]) / self.radius[hit_index]
            if self.object_type[hit_index] == CUBE_TYPE:
                n = cube_normal(p, self.center[hit_index])
            front_facing = is_front_facing(ray_direction, n)
            n = n if front_facing else -n

        return hit_anything, p, n, front_facing, hit_index

    @ti.func
    def scatter(self, ray_direction, p, n, front_facing, index):
        ''' Get the scattered direction for a ray hitting an object '''
        return self.materials.scatter(index, ray_direction, p, n, front_facing)
