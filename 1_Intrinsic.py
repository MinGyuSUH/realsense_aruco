import pyrealsense2 as rs
import numpy as np
import yaml

def save_realsense_intrinsics_to_yaml(yaml_path: str):
    # 1) RealSense 컬러 스트림에서 intrinsics 얻기
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)

    try:
        color_stream = profile.get_stream(rs.stream.color)
        intr = color_stream.as_video_stream_profile().get_intrinsics()

        # 2) 카메라 매트릭스 & 왜곡계수 구성
        camera_matrix = np.array([
            [intr.fx, 0.0,    intr.ppx],
            [0.0,     intr.fy, intr.ppy],
            [0.0,     0.0,    1.0]
        ], dtype=float)

        # RealSense intr.coeffs는 Brown–Conrady(일반적으로 k1,k2,p1,p2,k3)
        dist_coeff = np.array(intr.coeffs, dtype=float).tolist()

        # 3) YAML로 저장 (네가 쓰는 로더에 맞춰 키 이름 고정)
        data = {
            "camera_matrix": camera_matrix.tolist(),
            "dist_coeff": dist_coeff
        }

        with open(yaml_path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)

        print(f"[OK] Saved intrinsics to {yaml_path}")
        print("Camera Matrix:\n", camera_matrix)
        print("Distortion Coefficients:\n", np.array(dist_coeff))

    finally:
        pipeline.stop()

if __name__ == "__main__":
    save_realsense_intrinsics_to_yaml("camIntrinsic.yaml")
