import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import message_filters
import tf2_ros
from tf2_ros import TransformException
from geometry_msgs.msg import PointStamped
import tf2_geometry_msgs
from vision_msgs.msg import BoundingBox2DArray
import numpy as np

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_node')
        
        
        self.lidar_sub = message_filters.Subscriber(
            self, PointCloud2, '/point_cloud_A'
        )
        self.bbox_sub = message_filters.Subscriber(
            self, BoundingBox2DArray, '/yolo/bounding_boxes'
        )
        
      
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.lidar_sub, self.bbox_sub], queue_size=10, slop=0.1
        )
        self.ts.registerCallback(self.fusion_callback)
        
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera_info',
            self.camera_info_callback,
            10
        )
        
        self.publisher_objects_3d = self.create_publisher(
            PointCloud2, '/objects_3d', 10
        )
        
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.camera_model_ready = False

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.get_logger().info("Sensor Fusion Node Initialized. Waiting for CameraInfo, PointCloud2, and YOLO BBoxes...")

    def camera_info_callback(self, msg: CameraInfo):
        """CameraInfo 내부 파라미터 추출"""
        if not self.camera_model_ready:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.camera_model_ready = True
            self.get_logger().info(
                f"Successfully Extracted CameraInfo -> fx: {self.fx:.2f}, fy: {self.fy:.2f}, cx: {self.cx:.2f}, cy: {self.cy:.2f}"
            )

    def project_3d_to_2d(self, x: float, y: float, z: float):
        """3D 포인트 -> 2D 이미지 평면 좌표 투영"""
        if not self.camera_model_ready or z <= 0:
            return None
        u = (self.fx * (x / z)) + self.cx
        v = (self.fy * (y / z)) + self.cy
        return (int(u), int(v))

    def fusion_callback(self, cloud_msg, bbox_msg):
        if not self.camera_model_ready:
            self.get_logger().warn("Camera model not ready yet (Waiting for /camera_info)", throttle_duration_sec=2.0)
            return

        target_frame = "camera_A"
        source_frame = "OS1_A"

        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05)
            )
        except TransformException as ex:
            self.get_logger().warn(f"TF Transform Exception: {ex}", throttle_duration_sec=2.0)
            return

        try:
            point_list = list(pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True))
            if not point_list:
                return

            boxes = bbox_msg.boxes
            if not boxes:
                return

            for box_idx, box in enumerate(boxes):
                cx_box = box.center.position.x
                cy_box = box.center.position.y
                w = box.size_x
                h = box.size_y

                x_min = cx_box - (w / 2.0)
                x_max = cx_box + (w / 2.0)
                y_min = cy_box - (h / 2.0)
                y_max = cy_box + (h / 2.0)

                matched_points_3d = []

                for data in point_list:
                    lx, ly, lz = data[0], data[1], data[2]
                    if abs(lx) < 1e-3 and abs(ly) < 1e-3 and abs(lz) < 1e-3:
                        continue

                    pt_lidar = PointStamped()
                    pt_lidar.header = cloud_msg.header
                    pt_lidar.header.frame_id = source_frame
                    pt_lidar.point.x = float(lx)
                    pt_lidar.point.y = float(ly)
                    pt_lidar.point.z = float(lz)

                    try:
                        pt_camera = tf2_geometry_msgs.do_transform_point(pt_lidar, transform)
                        cx_val = pt_camera.point.x
                        cy_val = pt_camera.point.y
                        cz_val = pt_camera.point.z

                        pixel_coord = self.project_3d_to_2d(cx_val, cy_val, cz_val)
                        if pixel_coord:
                            u, v = pixel_coord
                            if x_min <= u <= x_max and y_min <= v <= y_max:
                                matched_points_3d.append([cx_val, cy_val, cz_val])

                    except Exception:
                        continue

                if not matched_points_3d:
                    continue

                target_points = np.array(matched_points_3d, dtype=np.float32)

                center_3d = np.median(target_points, axis=0)
                distance = np.linalg.norm(center_3d)

                self.get_logger().info(
                    f"Object #{box_idx} -> Center X: {center_3d[0]:.2f}, Y: {center_3d[1]:.2f}, Z: {center_3d[2]:.2f} | Dist: {distance:.2f}m"
                )

        except Exception as e:
            self.get_logger().error(f"Error in sensor fusion callback: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = SensorFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()