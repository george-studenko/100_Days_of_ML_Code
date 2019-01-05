# Let’s load a sample CT scan using the volread function in the imageio module, which takes a directory
# as argument and assembles all 'DICOM' (Digital Imaging Communication and Storage) files in a series in
# a NumPy 3D array.

import imageio

vol_arr = imageio.volread(dirname, 'DICOM')
print(vol_arr.shape)

# (256, 256, 50)

# Also in this case, the layout is different from what PyTorch expects. imageio outputs a W x H x D
# array, with no channel information. So we’ll first have to transpose and then make room for the
# channel dimension using unsqueeze:

vol = torch.from_numpy(col_arr).float()
vol = torch.transpose(vol, 0, 2)
vol = torch.unsqueeze(vol, 0)

vol.shape

torch.Size([1, 1, 50, 256, 256])

# At this point we could assemble a 5D dataset by stacking multiple volumes along the batch direction
