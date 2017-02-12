
#pragma once

namespace caffepro {
	namespace data_utils {
		enum crop_type : int {
			CropType_Random = 1,
			CropType_10View = 2,
			CropType_Center = 3,
			CropType_18View = 4,
			CropType_MultiCrop = 5,
			CropType_PropWarp = 6,
			CropType_FullView = 7
		};

		enum crop_position : int {
			CropPosition_Begin = 0,
			CropPosition_LeftTop,
			CropPosition_RightTop,
			CropPosition_LeftBottom,
			CropPosition_RightBottom,
			CropPosition_Center1,
			CropPosition_TopMiddle,
			CropPosition_RightMiddle,
			CropPosition_BottomMiddle,
			CropPosition_LeftMiddle,
			CropPosition_Center2,
			CropPosition_Extra1,
			CropPosition_Extra2,
			CropPosition_Extra3,
			CropPosition_Extra4,
			CropPosition_Extra5,
			CropPosition_Extra6,
			CropPosition_Extra7,
			CropPosition_Extra8,
			CropPosition_End
		};
	}
}
