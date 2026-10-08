"""lidar2grid: from airborne LiDAR to an electricity network model."""

# Internal label scheme. Loaders map dataset-specific codes (ASPRS, ECLAIR, DALES) onto these.
GROUND, VEGETATION, BUILDING, WIRE, POLE, OTHER = range(6)
CLASS_NAMES = ["ground", "vegetation", "building", "wire", "pole", "other"]
