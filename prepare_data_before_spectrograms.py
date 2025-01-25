# dividing for Class0 and Class1
from classify_audio import organize_audio

organize_audio("data/clean")
organize_audio("data/ipad_balcony1")
organize_audio("data/ipad_bedroom1")
organize_audio("data/ipad_confroom1")
organize_audio("data/ipad_confroom2")
organize_audio("data/ipad_livingroom1")
organize_audio("data/ipad_office1")
organize_audio("data/ipad_office2")
organize_audio("data/ipadflat_confroom1")
organize_audio("data/ipadflat_office1")
organize_audio("data/iphone_balcony1")
organize_audio("data/iphone_bedroom1")
organize_audio("data/iphone_livingroom1")


# removing silence from folders data/Class0 and data/Class1
from clear_folder import remove_silence_from_folder

remove_silence_from_folder('data/Class0')
remove_silence_from_folder('data/Class1')

# splitting cleared files into 3s segments
from split_all_files import split_all_files

split_all_files("data/class_0_cleared")
split_all_files("data/class_1_cleared")
