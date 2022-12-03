import taichi as ti
from vector import *
import ray
from time import time
from hittable import World, Sphere, Cube
from camera import Camera
from material import *
import math
import random
from taichi.math import vec3


# switch to cpu if needed
ti.init(arch=ti.gpu)


@ti.func
def get_background(dir):
    ''' Returns the background color for a given direction vector '''
    unit_direction = dir.normalized()
    t = 0.5 * (unit_direction[1] + 1.0)
    return (1.0 - t) * WHITE + t * BLUE


if __name__ == '__main__':
    # image data
    aspect_ratio = 3.0 / 2.0
    image_width = 1200
    image_height = int(image_width / aspect_ratio)
    rays = ray.Rays(image_width, image_height)
    pixels = ti.Vector.field(3, dtype=float)
    final_pixels = ti.Vector.field(3, dtype=float)
    attenuation_temp = ti.Vector.field(3, dtype=float)
    dir_temp = ti.Vector.field(3, dtype=float)
    sample_count = ti.field(dtype=ti.i32)
    needs_sample = ti.field(dtype=ti.i32)
    ti.root.dense(ti.ij,
                  (image_width, image_height)).place(
                    pixels, sample_count,
                    needs_sample, final_pixels,
                    attenuation_temp, dir_temp
                )

    samples_per_pixel = 512
    max_depth = 16

    # materials
    mat_ground = Lambert([0.5, 0.5, 0.5])
    mat2 = Lambert([0.4, 0.2, 0.2])
    mat1 = Dielectric(1.5)
    mat3 = Metal([0.7, 0.6, 0.5], 0.0)

    # world
    R = math.cos(math.pi / 4.0)
    world = World()
    world.add(Sphere([0.0, -1000, 0], 1000.0, mat_ground))

    static_point = Point(4.0, 0.2, 0.0)
    for a in range(-11, 11):
        for b in range(-11, 11):
            choose_mat = random.random()
            center = Point(a + 0.9 * random.random(), 0.2,
                           b + 0.9 * random.random())

            if (center - static_point).norm() > 0.9:
                if choose_mat < 0.8:
                    # diffuse
                    mat = Lambert(
                        Color(random.random(), random.random(),
                              random.random())**2)
                elif choose_mat < 0.95:
                    # metal
                    mat = Metal(
                        Color(random.random(), random.random(),
                              random.random()) * 0.5 + 0.5,
                        random.random() * 0.5)
                else:
                    mat = Dielectric(1.5)

            world.add(Sphere(center, 0.2, mat))

    world.add(Sphere([0.0, 1.0, 0.0], 1.0, mat1))
    world.add(Sphere([-4.0, 1.0, 0.0], 1.0, mat2))
    world.add(Cube([4.0, 1.0, 0.0], 1.0, mat3))
    world.commit()

    # camera
    vfrom = Point(13.0, 2.0, 3.0)
    at = Point(0.0, 0.0, 0.0)
    up = Vector(0.0, 1.0, 0.0)
    focus_dist = 10.0
    aperture = 0.1
    cam = Camera(vfrom, at, up, 20.0, aspect_ratio, aperture, focus_dist)

    start_attenuation = Vector(1.0, 1.0, 1.0)
    initial = True

    max_depth = 50

    @ti.kernel
    def render_complete():
        for x, y in pixels:
            pdf = attenuation_temp[x, y]
            ray_dir = dir_temp[x, y]
            pixels[x, y] += pdf * ray_dir

    @ti.kernel
    def finish(cnt: int):
        for x, y in pixels:
            pdf = attenuation_temp[x, y]
            ray_dir = dir_temp[x, y]
            pixels[x, y] += pdf * ray_dir
            final_pixels[x, y] = ti.sqrt(pixels[x, y] / cnt)

    @ti.kernel
    def wavefront_initial():
        for x, y in pixels:
            sample_count[x, y] = 0
            needs_sample[x, y] = 1

    @ti.kernel
    def render():
        ''' Loops over pixels
            for each pixel:
                generate ray if needed
                intersect scene with ray
                if miss or last bounce sample backgound
            return pixels that hit max samples
        '''
        # num_completed = 0
        for x, y in pixels:
            # if sample_count[x, y] == samples_per_pixel:
            #     continue

            # gen sample
            ray_org = Point(0.0, 0.0, 0.0)
            ray_dir = Vector(0.0, 0.0, 0.0)
            # depth = max_depth
            pdf = start_attenuation

            u = (x + ti.random()) / (image_width - 1)
            v = (y + ti.random()) / (image_height - 1)
            ray_org, ray_dir = cam.get_ray(u, v)
            # rays.set(x, y, ray_org, ray_dir, depth, pdf)

            d = 0
            while True:
                d += 1
                # intersect
                hit, p, n, front_facing, index = world.hit_all(ray_org, ray_dir)
                if hit:
                    reflected, out_origin, out_direction, attenuation = world.materials.scatter(
                        index, ray_dir, p, n, front_facing)
                    if reflected:
                        pdf *= attenuation
                    else:
                        attenuation_temp[x, y] = vec3(0.0)
                        break

                    ray_org = out_origin
                    ray_dir = out_direction
                    
                if not hit or d == max_depth:
                    attenuation_temp[x, y] = pdf
                    dir_temp[x, y] = get_background(ray_dir)
                    # pixels[x, y] += pdf * get_background(ray_dir)
                    break

    window = ti.ui.Window("Taichi RayTracer", (image_width, image_height), vsync=False)
    canvas = window.get_canvas()
    gui = window.get_gui()
    d = 0
    while window.running:
        d += 1
        # wavefront_initial()
        render()
        # render_complete()
        finish(d)
        canvas.set_image(final_pixels)
        window.show()

