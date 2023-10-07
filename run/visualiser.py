from matplotlib import pyplot
from matplotlib.patches import Rectangle, Circle
import torch

class Visualiser:

	def __init__(self, visible=False):
		self.visible = visible
		self.reset()

	def reset(self):
		pyplot.cla()
		self.fig = pyplot.gcf()
		self.ax = pyplot.gca()


	def show_objects(self, x, agent_pos=torch.zeros(2), **kwargs):
		x = x.detach()
		for i in range(x.shape[0]):
			colour = torch.abs(x[i,:3])
			radius = torch.abs(x[i,3]) / 8
			pos = (x[i,4:6] - agent_pos) * 1
			self.create_shape(colour=colour, pos=pos, radius=radius, type="circle", **kwargs)


	def highlight_agent(self, agent_pos):
		self.create_shape(colour="blue", pos=agent_pos, radius=0.1, type="box")


	def create_shape(self, colour, pos, radius=None, pos2=None, type="circle", **kwargs):
		def rgb2hex(rgb, bounds=1):
			rgb = torch.clamp(rgb, 0, bounds) / bounds
			rgbint = (rgb * 255).int()
			return "#{:02x}{:02x}{:02x}".format(rgbint[0], rgbint[1], rgbint[2])
		colour = rgb2hex(colour) if not isinstance(colour, str) else colour
		if pos2 is not None:
			if type == "line":
				self.ax.arrow(pos[0], pos[1], pos2[0]-pos[0], pos2[1]-pos[1], color=colour, **kwargs)
		elif radius is not None:
			if type == "circle":
				shape = Circle((pos[0], pos[1]), radius=radius, facecolor=colour, edgecolor="black", linewidth=4, **kwargs)
			elif type == "box":
				shape = Rectangle((pos[0]-radius, pos[1]-radius), width=2*radius, height=2*radius, facecolor=colour, edgecolor="black", **kwargs)
			self.ax.add_patch(shape)

	def render(self, lim=None, grid=False, axis=False, margin=0.):
		self.ax.axis("equal")
		if isinstance(lim, list) and len(lim) == 4:
			self.ax.set_xlim(left=lim[0], right=lim[1])
			self.ax.set_ylim(bottom=lim[2], top=lim[3])
		elif isinstance(lim, list) and len(lim) == 2:
			self.ax.set_xlim(left=lim[0], right=lim[1])
			self.ax.set_ylim(bottom=lim[0], top=lim[1])
		elif isinstance(lim, float):
			self.ax.set_xlim(left=-lim, right=lim)
			self.ax.set_ylim(bottom=-lim, top=lim)
		self.fig.subplots_adjust(bottom=margin, top=1-margin, left=margin, right=1-margin)
		if not axis:
			for tick in self.ax.xaxis.get_major_ticks() + self.ax.yaxis.get_major_ticks():
				tick.tick1line.set_visible(False)
				tick.tick2line.set_visible(False)
				tick.label1.set_visible(False)
				tick.label2.set_visible(False)
		if grid:
			self.ax.grid()
			self.ax.set_axisbelow(True)
		if self.visible:
			self.fig.show()
		return self.fig

	def save(self, path):
		self.fig.savefig(path, bbox_inches="tight")

	def close(self):
		pyplot.close()

if __name__ == "__main__":
	vis = Visualiser(visible=True)
	x = torch.randn(4,6)
	x_noisy = x + torch.randn(4,6) / 10
	vis.show_objects(x_noisy, alpha=0.3, linestyle="dashed")
	vis.show_objects(x)
	vis.render(grid=True)