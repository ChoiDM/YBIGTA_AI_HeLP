import numpy as np

def change_values(expanded_mask, center_index, size):
    x_idx, y_idx = center_index

    for i in range(-size, size+1):
        for j in range(-size, size+1):
            expanded_mask[x_idx+i, y_idx+j] = 1


def normalization(img_array, mask_array, size):

    n_target = 0
    sum_values = 0

    for z_idx in range(mask_array.shape[-1]):

        mask_slice = mask_array[:, :, z_idx]
        expanded_mask = np.zeros_like(mask_slice)

        # If slice has infarct mask
        if len(np.unique(mask_slice)) > 1:
            
            for x_idx in range(mask_slice.shape[0]):
                for y_idx in range(mask_slice.shape[1]):

                    # If pixel value is 1.0 (infarct mask)
                    if (mask_slice[x_idx, y_idx] != 0.0) and (x_idx+1 < mask_slice.shape[0]) and (y_idx+1 < mask_slice.shape[1]):
                        change_values(expanded_mask, [x_idx, y_idx], size)
            
            expanded_mask[mask_slice != 0] = 0
            
            target_values = img_array[:, :, z_idx][mask_slice != 0]

            n_target += len(target_values)
            sum_values += sum(target_values)


    if n_target > 0:
        norm_factor = (sum_values // n_target)
        return img_array / norm_factor
    
    else:
        print(">>> Unable to do normalization : No value 1 in mask array.")
        return img_array
