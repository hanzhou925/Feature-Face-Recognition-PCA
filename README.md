# Feature Face Recognition (Eigenface)
**1. Background:**

Eigenface is the first effective facial recognition method, obtained by performing principal component analysis (PCA) on a large set of images depicting different faces. The dataset of this experiment is based on ORL database (http://cam-orl.co.uk/facedatabase.html)

MNIST is a handwritten digit data set. The training set contains 60,000 handwritten digits, and the test set contains 10,000 handwritten digits, with a total of 10 categories.
 
 
**2. Procedure:**

In the process of model training, the average face is firstly calculated according to the test data, and then the first K feature faces are saved to recognize the test face. In addition, for any given face image, the K feature faces can be used to reconstruct the original image.

![1632815962998_6EBE061B-A27D-4707-BD29-75C33DB8D86B](https://user-images.githubusercontent.com/91419621/135047346-03c9a500-3986-48d6-97ae-43d56deec701.png)
------
![1632816011211_3FABB6BF-3610-4a76-98FA-710784E97F6A](https://user-images.githubusercontent.com/91419621/135047455-8dd3d4c7-4fd8-4788-a912-d3a7f0cc8cb4.png)
------
![1632816043914_8C2ABE3C-CE66-4a60-94B6-46F49652B92F](https://user-images.githubusercontent.com/91419621/135047540-76c6edb7-f736-4546-8573-d6c412e801b9.png)



