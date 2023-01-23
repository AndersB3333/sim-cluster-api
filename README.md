# sim-cluster-api
This is a program that simulates n golf shots based on the previous golf shots entered. The purpose of this program is to give the user a model of your their overall shot pattern based on their previous golf shots. 

# Inputs
The program takes in an JSON array length of 11 * 11 + (preferred hand) + number of simulated shots. The preferred hand is a binary value that is 0 if right handed, and 1 if left-handed. This information will be important for calculating the probabilites of where the shots will land.

![image](https://user-images.githubusercontent.com/94805074/213934205-8b6ca03f-23e5-4f28-a561-2db5414abe36.png)

The image represents a discrete example of how the json array is structured. The values goes from top-left (188.1 - 192.0 yard shot, that went 22.5-27.5 yard left) to top-right, then proceeds to the next line, and repeat. The total amount of shots has to be > 9, to make sure there are enough observations that the model can make a simulation out of.

# Motivation
Knowing a golfer's shot pattern might help the golfer make smarter decisions that will reduce the chances of high scores on certain holes. From personal experience, hobby golfers does not know their shot pattern well, and if they did they would make different choices on the course. A simple example would be a golfer planning on whether she should try to attack the pin from 170 yards out or not when there's a tricky bunker tucked 8 yards to the right of the pin. If the golfer's actual shot pattern was that 65% of the shots from this distance ends up > 8 yards to the right of the pin, then a rational decision would be to pick a more conservative target to avoid the probabilites of ending up in the bunker. As this example displays, a person that knows their shot pattern might save several shots each round, and this program attempts to do so. 

# Calculations
The program converts the golf shots data to cartesian coordinates to more conveniently calculate the distance and direction between each of the shots. From here the boxes from picture 1 (referred in this program as "bins") that has a > 0 frequency are referred to as strong bins. The strong bins will have a stronger probability to occur in the future since the golfer has already hit in that area. A golfer does usually have some common misses, and K-means clustering attempts to segment the good shots and the bad shots into different clusters. The number of clusters were derived from rounding the square root of the strong bins in the dataset. 
```math
\mu \approx \sqrt{|strong bins|}
```
After the clusters are assigned, the probabilites of the values inside each clusters increase with the cardinality of the clusters $\hat{x_{i}} = x_{i} * |cluster_{j}| $. 

The dataset needs a center point to calculate the probabilites of the values. This were derived from a simple midpoint calculation of cartesian coordinates that will not be shown here (the function is called centroid_calc in the code). 

The competency of the golfer is calculated by summing up the values of the frequency where the golfer hit in regards to the point system provided below. 

![image](https://user-images.githubusercontent.com/94805074/213936916-a2605cc4-1155-4c29-b74c-47d9af8ccf13.png)

Let points the number assigned to each bin in the diagram below, and denoted as lowercase sigma, and the result be theta:
```math
\theta = \frac{1}{N} \sum_{n=1}^{N} \sigma \hat{x_{i}}
```
The values that currently does have value 0 will be assigned $\frac{1}{10}\sum x_{i}$ to avoid multiplication by 0 in the main probability algorithm.

Distance denoted as $\rho$ between each coordinate is calculated by L2 norm, specifically Frobenius' formula.

Direction, denoted in the formula as $\lambda$ is provided by the diagram below. This diagram is mirrored for left-handed players, therefore, it's important to have this correct as an input.

The final probability of each value $\widecheck{x}$ is denoted below:
```math
\widecheck{x} = \sum_{j = 1}^{\mu} \frac{\hat{x_{i}} \lambda }{(1 + \rho)^{3}} + |15 - \rho| \hat{x_{i}} \sigma
```
If the value had a frequency > 0 in the original dataset, the final value will be divided by 2 as a buffer (since these values would be too large, and probability of the zero bins will be too low).

The probability distribution output of the values in list 1 looks like this:

<img width="353" alt="image" src="https://user-images.githubusercontent.com/94805074/213938579-7d98035b-e43a-48e8-90dc-3895164030d2.png">

This heatmap displays the relationship between the probability of each shot ending up in that specific bin.

Through NumPy's uniformily distributed random number generator, these values are assigned based on their respective sequence. So, for example, if the probability of bin 71 is: $f(\widecheck{x_{71}}) = p(45<= \widecheck{x_{71}} <= 50)$, then all values of the random number generator between 45 and 50 will reflect simulated shots entering this specific bin.

So, conducting the probability accumulation, the output looks like this:
<img width="347" alt="image" src="https://user-images.githubusercontent.com/94805074/213973638-2ed0595e-8b44-47cc-8117-026f19549e7d.png">










