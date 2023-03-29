# # %% chatgpt
import pybullet as p
import pybullet_data

# Connect to a PyBullet physics server
p.connect(p.GUI)

# Load the TurtleBot model
robot_urdf = "turtlebot.urdf"
robot_start_pos = [0, 0, 0.1]
robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
robot_id = p.loadURDF(robot_urdf, robot_start_pos, robot_start_orientation)

# Load the ground plane
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane_id = p.loadURDF("plane.urdf")

# Set up the simulation parameters
time_step = 1/240
max_simulation_time = 100
p.setTimeStep(time_step)
p.setGravity(0, 0, -9.81)
p.setRealTimeSimulation(0)

# Main simulation loop
for i in range(int(max_simulation_time/time_step)):
    # Get the current position and orientation of the TurtleBot
    robot_pos, robot_orn = p.getBasePositionAndOrientation(robot_id)

    # Move the TurtleBot forward
    p.setJointMotorControl2(robot_id, 0, p.VELOCITY_CONTROL, targetVelocity=1, force=1)

    # Step the simulation
    p.stepSimulation()

# Clean up
p.disconnect()


# # %%
# import pybullet as p
# import time
# p.connect(p.GUI)
# offset = [0,0,0]

# turtle = p.loadURDF("turtlebot.urdf",offset)
# plane = p.loadURDF("plane.urdf")
# p.setRealTimeSimulation(1)

# for j in range (p.getNumJoints(turtle)):
# 	print(p.getJointInfo(turtle,j))
# forward=0
# turn=0

# while (1):
    
# 	p.setGravity(0,0,-10)
# 	time.sleep(1./240.)
# 	keys = p.getKeyboardEvents()
# 	leftWheelVelocity=0
# 	rightWheelVelocity=0
# 	speed=10

#     for k,v in keys.items():
        
#         if (k == p.B3G_RIGHT_ARROW and (v&p.KEY_WAS_TRIGGERED)):
#             turn = -0.5
#         if (k == p.B3G_RIGHT_ARROW and (v&p.KEY_WAS_RELEASED)):
#             turn = 0
#         if (k == p.B3G_LEFT_ARROW and (v&p.KEY_WAS_TRIGGERED)):
#             turn = 0.5
#         if (k == p.B3G_LEFT_ARROW and (v&p.KEY_WAS_RELEASED)):
#             turn = 0

#         if (k == p.B3G_UP_ARROW and (v&p.KEY_WAS_TRIGGERED)):
#             forward=1
#         if (k == p.B3G_UP_ARROW and (v&p.KEY_WAS_RELEASED)):
#             forward=0
#         if (k == p.B3G_DOWN_ARROW and (v&p.KEY_WAS_TRIGGERED)):
#             forward=-1
#         if (k == p.B3G_DOWN_ARROW and (v&p.KEY_WAS_RELEASED)):
#             forward=0
                        
#     rightWheelVelocity += (forward+turn)*speed
#     leftWheelVelocity += (forward-turn)*speed

#     p.setJointMotorControl2(turtle,0,p.VELOCITY_CONTROL,targetVelocity=leftWheelVelocity,force=1000)
#     p.setJointMotorControl2(turtle,1,p.VELOCITY_CONTROL,targetVelocity=rightWheelVelocity,force=1000)

