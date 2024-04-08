from dataset import PlantDataset

dataset = PlantDataset(root="./dataset", mode="train")
a = dataset.__getitem__(0)

print("image tensors (shape: ",a["image"].shape,")")
print(a["image"])

print("data tensors (shape: ",a["data"].shape,")")
print(a["data"])

print("label tensors (shape: ",a["label"].shape,")")
print(a["label"])