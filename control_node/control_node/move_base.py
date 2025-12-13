import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class CmdVel(Node):

    def __init__(self):
        super().__init__('cmd_pub')

        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.publish_vel)

    def publish_vel(self):
        msg = Twist()
        msg.linear.x = float(0)
        msg.linear.y = float(0.5)
        msg.angular.z = float(0)

        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CmdVel()

    rclpy.spin(node=node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
