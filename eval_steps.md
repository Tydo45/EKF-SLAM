1. Each point should look at all the true points it's near and choose it's nearest neighbor
2. calculate how far it is from the true point  

There needs to be either an accepted radius for if a point is close enough to be true... or there needs to be some sort of an equation based on how close the point is to the actual.  
For example we could use y=-x+100, and x = distance from closest point  
then points close are close to 100% accuracy  
points far away are close to 0  
Of course it should be an equation that approaches 0 but never crosses it  

3. Develop a way to provide a value to a distance from a point

# value from a point
Use:  
y=1/((x*0.01/b)+0.01)  
b is how far away a point can be to reach at least 50% accuracy
