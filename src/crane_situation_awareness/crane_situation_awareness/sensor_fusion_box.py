import numpy as np 

def extract_bbox_points_and_calcuate_center(points: np.ndarray, bbox: list):
    if points is None or len(points) == 0:
        print("warning: empty")
        return np.empty((0,3)), None
    
    if bbox is None or len(bbox) != 6:
        print("warning: invalid bbox")
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

#test
if __name__ == "__main__":
    incoming_points = np.array([
        [1.2, 0.5, -0.1], 
        [-5.0, 2.0, 1.0],
        [0.8, -0.2, 0.3],
        [1.5, 1.1, -0.2],
        [10.0, 10.0, 10.0]
    ])
    
    incoming_bbox = [0.0, -1.0, -1.0, 2.0, 2.0, 1.0]
    
    points_in_box, object_position = extract_bbox_points_and_calcuate_center(incoming_points, incoming_bbox)

   

    print("test result")
    print(f"Number of points inside the bounding box: {len(points_in_box)}")
    print(f"Points inside the bounding box:\n{points_in_box}")
    print(f"Calculated center of the points inside the bounding box: {object_position}")
