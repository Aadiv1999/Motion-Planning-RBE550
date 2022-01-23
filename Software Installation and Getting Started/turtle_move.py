from turtle import *

color('red', 'yellow')
begin_fill()

# stop drawing
penup()

# move to initial position to center of the screen
left(180)
forward(250)
right(180-36)

# get the start position
start = pos()

# start drawing
pendown()
while True:
    forward(500)
    right(180-36)

    if abs(pos()-start) < 1:
        break

end_fill()
done()