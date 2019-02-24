import numpy as np



def get_center(mask_array):
    x_values, y_values, z_values = np.where(mask_array > 0.5)
    
    # Mask has 0 or 1
    if len(x_values) > 0:
        x_center = int(np.mean(np.array(x_values)))
        y_center = int(np.mean(np.array(y_values)))
        z_center = int(np.mean(np.array(z_values)))

        return [x_center, y_center, z_center]
    
    # Mask only has 0
    else:
        return None

def cube_indexing(center_point, cube_size):
    half_size = cube_size // 2

    # When Cube Size is Even Number
    if (half_size * 2) == cube_size:
        indexing = [center_point - half_size, center_point + half_size]
    
    # When Cube Size is Odd Number
    else: 
        indexing = [center_point - half_size, center_point + half_size + 1]
    
    return indexing

    
def img2cube(img_array, cube_center, cube_shape):
    cube_array = np.copy(img_array)

    x_size, y_size, z_size = np.array(cube_shape)

    if not (x_size > img_array.shape[0] or y_size > img_array.shape[1] or z_size > img_array.shape[2]):
        x_center, y_center, z_center = cube_center
        
        x_index = cube_indexing(x_center, x_size)
        y_index = cube_indexing(y_center, y_size)
        z_index = cube_indexing(z_center, z_size)

        cube_array = cube_array[x_index[0] : x_index[1],
                                y_index[0] : y_index[1],
                                z_index[0] : z_index[1]]
        
        return cube_array
    
    else:
        print("Please reset the shape of cube (Cube is out of image)")
        return None
