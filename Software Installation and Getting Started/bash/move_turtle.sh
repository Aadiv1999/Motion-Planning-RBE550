rosservice call turtle1/teleport_absolute 5.5 5.5 0
rosservice call turtle2/teleport_absolute 5.5 5.5 0
rosservice call turtle3/teleport_absolute 5.5 5.5 0
rosservice call turtle4/teleport_absolute 5.5 5.5 0
rosservice call turtle5/teleport_absolute 5.5 5.5 0

rosservice call /kill "turtle1"
rosservice call /kill "turtle2"
rosservice call /kill "turtle3"
rosservice call /kill "turtle4"
rosservice call /kill "turtle5"