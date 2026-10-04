"""Render recorded EngineAI T800 policy states with the official MuJoCo model."""

from __future__ import annotations

import argparse
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="T800 learned policy")
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera-distance", type=float, default=2.75,
                        help="Display-only camera distance; recorded states are unchanged.")
    args = parser.parse_args()
    if not np.isfinite(args.camera_distance) or args.camera_distance <= 0:
        parser.error("--camera-distance must be finite and positive")
    return args


def build_scene_xml(source: Path, target: Path) -> None:
    tree = ET.parse(source)
    root = tree.getroot()
    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        raise ValueError("T800 XML is missing asset or worldbody")
    ET.SubElement(
        asset,
        "texture",
        name="policy_ground_tex",
        type="2d",
        builtin="checker",
        rgb1="0.18 0.20 0.23",
        rgb2="0.48 0.51 0.55",
        width="512",
        height="512",
    )
    ET.SubElement(
        asset,
        "material",
        name="policy_ground_mat",
        texture="policy_ground_tex",
        # On an infinite plane texture repeat is per metre, not per finite
        # rectangle; keep tiles large enough to judge foot/ground motion.
        texrepeat="1.5 1.5",
        reflectance="0.02",
        shininess="0.10",
    )
    plane = ET.Element(
        "geom",
        name="policy_ground",
        type="plane",
        # MuJoCo planes collide infinitely; zero XY extents also render the
        # plane infinitely, so locomotion cannot leave the visible floor.
        size="0 0 0.1",
        material="policy_ground_mat",
        friction="1.0 0.005 0.0001",
        contype="1",
        conaffinity="1",
    )
    worldbody.insert(0, plane)
    # The original fixed spot lights leave a moving robot in darkness. Use
    # MuJoCo's camera-mounted headlight and directional fill; neither depends
    # on world position. Recorded robot/ground poses are not changed.
    ET.SubElement(worldbody, "light", name="policy_directional_fill",
                  directional="true", pos="0 0 5", dir="-0.3 0.2 -1",
                  diffuse="0.40 0.40 0.40", specular="0.05 0.05 0.05",
                  castshadow="false")
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    global_cfg = visual.find("global")
    if global_cfg is None:
        global_cfg = ET.SubElement(visual, "global")
    global_cfg.set("offwidth", "960")
    global_cfg.set("offheight", "720")
    headlight = visual.find("headlight")
    if headlight is None:
        headlight = ET.SubElement(visual, "headlight")
    headlight.attrib.update(active="1", ambient="0.35 0.35 0.35",
                            diffuse="0.50 0.50 0.50", specular="0.05 0.05 0.05")
    tree.write(target, encoding="utf-8", xml_declaration=True)


def main() -> None:
    args = parse_args()
    if args.output.exists() or args.output.with_name(args.output.stem + "_contact_sheet.png").exists():
        raise FileExistsError("Refusing to overwrite rendered evidence")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with np.load(args.rollout, allow_pickle=False) as data:
        joint_pos = data["joint_pos"].copy()
        joint_names = [str(name) for name in data["joint_names"]]
        root_pos = data["root_pos"].copy()
        if "root_quat_wxyz" in data:
            root_quat = data["root_quat_wxyz"].copy()
        else:
            # EngineAI recording's root_quat is Isaac/MuJoCo wxyz.
            root_quat = data["root_quat"].copy()
        if "quaternion_order" in data and str(data["quaternion_order"].item()) != "wxyz":
            raise ValueError("Renderer expects explicitly recorded wxyz quaternions")

    if not (len(joint_pos) == len(root_pos) == len(root_quat)):
        raise ValueError("Rollout arrays have inconsistent frame counts")
    if not np.isfinite(joint_pos).all() or not np.isfinite(root_pos).all() or not np.isfinite(root_quat).all():
        raise ValueError("Rollout contains non-finite state values")

    with tempfile.NamedTemporaryFile(
        prefix="serial_t800_policy_", suffix=".xml", dir=args.xml.parent, delete=False
    ) as temp_file:
        temp_xml = Path(temp_file.name)
    try:
        build_scene_xml(args.xml, temp_xml)
        model = mujoco.MjModel.from_xml_path(str(temp_xml))
        data = mujoco.MjData(model)
        qpos_addresses = {}
        for name in joint_names:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"Official XML is missing rollout joint: {name}")
            qpos_addresses[name] = int(model.jnt_qposadr[joint_id])

        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.distance = args.camera_distance
        camera.azimuth = 135.0
        camera.elevation = -14.0
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)
        writer = imageio.get_writer(
            args.output,
            fps=args.fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
        )
        sheet_ids = set(np.linspace(0, len(joint_pos) - 1, 12, dtype=int).tolist())
        sheet_frames: list[tuple[int, np.ndarray]] = []
        try:
            for frame_index in range(len(joint_pos)):
                data.qpos[:3] = root_pos[frame_index]
                data.qpos[3:7] = root_quat[frame_index]
                for dof_index, name in enumerate(joint_names):
                    data.qpos[qpos_addresses[name]] = joint_pos[frame_index, dof_index]
                mujoco.mj_forward(model, data)
                camera.lookat[:] = root_pos[frame_index] + np.array([0.0, 0.0, 0.10])
                renderer.update_scene(data, camera=camera)
                rgb = renderer.render().copy()
                image = Image.fromarray(rgb)
                draw = ImageDraw.Draw(image)
                draw.rectangle((14, 12, 570, 48), fill=(0, 0, 0, 150))
                draw.text(
                    (24, 21),
                    f"{args.label}  {frame_index / args.fps:05.2f}s",
                    fill=(255, 255, 255),
                )
                rgb = np.asarray(image)
                writer.append_data(rgb)
                if frame_index in sheet_ids:
                    sheet_frames.append((frame_index, rgb.copy()))
        finally:
            writer.close()
            renderer.close()

        tile_width, tile_height = 480, 360
        sheet = Image.new("RGB", (tile_width * 4, tile_height * 3), (20, 20, 20))
        for tile_index, (frame_index, rgb) in enumerate(sheet_frames):
            tile = Image.fromarray(rgb).resize((tile_width, tile_height), Image.Resampling.LANCZOS)
            tile_draw = ImageDraw.Draw(tile)
            tile_draw.text((12, 12), f"{frame_index / args.fps:.1f}s", fill=(255, 255, 255))
            sheet.paste(tile, ((tile_index % 4) * tile_width, (tile_index // 4) * tile_height))
        sheet_path = args.output.with_name(args.output.stem + "_contact_sheet.png")
        sheet.save(sheet_path)
        print(f"Rendered {len(joint_pos)} policy frames to {args.output}")
        print(f"Saved contact sheet to {sheet_path}")
    finally:
        temp_xml.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
