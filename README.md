# Programming's project
## Prediction of the normalized price of a used device

In the digital age we live in, personal devices have become an essential tool. The rapid and constant technological advancements have led to the continuous introduction in the market of new devices, such as smartphones, tablets, smartwhatches, smart tvs and other. As a result, the used industry has become increasingly relevant, offering a more affordable option for those looking to own a quality device at a lower price.

Choosing a price for a used device may be a complex task, that\'s why for my programming project I decided to prepare a statistical analysis on predicting the normalized price of a used device. To estimate the normalized price of a used phone. To achieve this, I decided to create a linear regression model for which I chose the most influent variables through forward selection. The statistical analysis will be based on a dataset made of 3454 samples (devices) and 15 variables that includes detailed information about used phones, such as technical specifications and design features. The variables I took in account are:

- **device_brand** is the name of manufacturing brand;
- **os** is the operating system on which the device runs;
- **screen_size** is the size of the screen in cm;
- **4g** is a string declaring whether 4G is available or not;
- **5g** is a string declaring whether 5G is available or not;
- **rear_camera_mp** is the resolution of the rear camera in megapixels;
- **front_camera_mp** is the resolution of the front camera in megapixels;
- **internal_memory** is the amount of internal memory (ROM) in GB;
- **ram** is the amount of RAM in GB;
- **battery** is the energy capacity of the device battery in mAh;
- **weight** is the weight of the device in grams;
- **release_year** is the year when the device model was released;
- **days_used** is the number of days the used/refurbished device has been used;
- **normalized_new_price** is the normalized price of a new device of the same model;
- **normalized_used_price** is the normalized price of the used/refurbished device.