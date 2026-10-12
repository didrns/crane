import numpy as np 

def extract_bbox_points_and_calcuate_center(points,bbox):
    if points is None or len(points) == 0:
        return np.empty((0,3)), None

    x_min, y_min, z_min, x_max, y_max, z_max = bbox


    mask = (
        (points[:, 0] >= x_min) & (points[:, 0]<= x_max) &
        (points[:, 1] >= y_min) & (points[:, 1]<= y_max) &
        (points[:, 2] >= z_min) & (points[:, 2]<= z_max) 

    )

    inside_points = points[mask]

    if len(inside_points) > 0:
        center_3d = np.median(inside_points, axis=0)
    else:
        center_3d = None
    
    return inside_points, center_3d

if __name__ == "__main__":
    np.random.seed(42)
    dummy_points = np.random.uniform(low=-10.0, high=10.0, size=(1000, 3))

    temp_bbox = [-2.0, -2.0, -1.0, 2.0, 2.0, 1.0]

    points_in_box, object_position = extract_bbox_points_and_calcuate_center(dummy_points, temp_bbox)

    print("test result")
    print(f"{len(points_in_box)} ")
    print(f"xyz: {object_position}")
    

