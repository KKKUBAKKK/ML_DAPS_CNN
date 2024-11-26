# dividing for class_0 and class_1
from classify_audio import classify_audio

# organize_audio("data/clean")
# organize_audio("data/ipad_balcony1")
# organize_audio("data/ipad_bedroom1")
# organize_audio("data/ipad_confroom1")
# organize_audio("data/ipad_confroom2")
# organize_audio("data/ipad_livingroom1")
# organize_audio("data/ipad_office1")
# organize_audio("data/ipad_office2")
# organize_audio("data/ipadflat_confroom1")
# organize_audio("data/ipadflat_office1")
# organize_audio("data/iphone_balcony1")
# organize_audio("data/iphone_bedroom1")
# organize_audio("data/iphone_livingroom1")


# removing silence from folders data/class_0 and data/class_1
from remove_silence_from_directory import remove_silence_from_directory

remove_silence_from_directory('./daps/data/audio/test/class_0')
remove_silence_from_directory('./daps/data/audio/test/class_1')

remove_silence_from_directory('./daps/data/audio/train/class_0')
remove_silence_from_directory('./daps/data/audio/train/class_1')

remove_silence_from_directory('./daps/data/audio/validation/class_0')
remove_silence_from_directory('./daps/data/audio/validation/class_1')

# splitting cleared files into 3s segments
from split_all_files import split_all_files

split_all_files("./daps/data/test/class_0_cleared")
split_all_files("./daps/data/test/class_1_cleared")

split_all_files("./daps/data/train/class_0_cleared")
split_all_files("./daps/data/train/class_1_cleared")

split_all_files("./daps/data/validation/class_0_cleared")
split_all_files("./daps/data/validation/class_1_cleared")
