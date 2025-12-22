import json
import os.path
import typing

import networkx as nx
import matplotlib.pyplot as plt
import random

from core import Config
from core.agent.action import ActionSequence

durations = {
	phase: 0
	for phase in [
		'market_state_copy',
		'agent_state_copy',
		'open_trades_copy',

		'prediction',
		'state_action_to_model_input',

		'promising_action_node',
		'random_state_node',

		'expected_transition_probability',
		'random_choice',
		'expected_transition_probability_norm',

		'get_expected_transition_probability_distribution',
		'prepare_dta_input',
		'predict',
		'prepare_dta_output',
		'get_cached',
		'cache_predictions',
		'get_expected_transition_probability_distribution__',
		'unique',
		'model_predictions',
		'__get_unique_rows',
		'hash',

		'state_comparison',
		'find_gap_index',

		'DeepReinforcementAgent._get_state_action_value',
		'super()._get_action_node_value',
		'for i in range(len(probability_distribution))',
		'trim',

		'state.get_agent_state().close_trades',
		'state.get_market_state().get_current_price',

		'select',
		'expand',
		'simulate',
		'backpropagate'

	]
}

iterations = {
	phase: 0
	for phase in [
		'open_trades_copy', "main_loop", 'prediction', 'cached_prediction', 'expected_transition_probability_nodes'
	]
}

possible_state_visits = []
valid_actions = []
prediction_inputs = []


def get_percentages(total):
	percentages = {}
	for phase, duration in durations.items():
		percentages[phase] = duration*100/total
	return percentages


def get_nodes(parent_node, depth=None, top=None, visited=False, include_root=False):
	children = parent_node.get_children()
	if visited:
		children = [node for node in children if node.visits > 0]

	inner_top = top
	if top is None:
		inner_top = len(children)

	if len(children) == 0:
		return []

	inner_top = min(inner_top, len(children))

	threshold = sorted([node.weight for node in children], reverse=True)[inner_top-1]

	if depth == 0:
		return children
	elif depth is not None:
		depth -= 1
	nodes = []
	for node in children:
		if node.node_type == 0 and node.weight < threshold:
			continue
		nodes += get_nodes(node, depth, top=top, visited=visited)
		nodes.append(node)

	if include_root:
		nodes.append(parent_node)

	return nodes


def get_max_depth(node):

	children = node.get_children()
	if len(children) == 0:
		return 1
	return max(
		[
			get_max_depth(child)
			for child in children
		]
	) + 1


def remove_duplicate(values):
	filtered_values = []
	for value in values:
		if value not in filtered_values:
			filtered_values.append(value)
	return filtered_values


def draw_test_graph():
	graph = nx.Graph()
	graph.add_edge("A", "B")
	graph.add_edge("A", "C")
	nx.draw(graph)
	plt.show()

def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):

	'''
	From Joel's answer at https://stackoverflow.com/a/29597209/2966723.
	Licensed under Creative Commons Attribution-Share Alike
	If the graph is a tree this will return the positions to plot this in a
	hierarchical layout.
	G: the graph (must be a tree)
	root: the root node of current branch
	- if the tree is directed and this is not given,
	  the root will be found and used
	- if the tree is directed and this is given, then
	  the positions will be just for the descendants of this node.
	- if the tree is undirected and not given,
	  then a random choice will be used.
	width: horizontal space allocated for this branch - avoids overlap with other branches
	vert_gap: gap between levels of hierarchy
	vert_loc: vertical location of root
	xcenter: horizontal location of root
	'''
	if not nx.is_tree(G):
		raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

	if root is None:
		if isinstance(G, nx.DiGraph):
			root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
		else:
			root = random.choice(list(G.nodes))

	def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):

		if pos is None:
			pos = {root:(xcenter,vert_loc)}
		else:
			pos[root] = (xcenter, vert_loc)
		children = list(G.neighbors(root))
		if not isinstance(G, nx.DiGraph) and parent is not None:
			children.remove(parent)
		if len(children)!=0:
			dx = width/len(children)
			nextx = xcenter - width/2 - dx/2
			for child in children:
				nextx += dx
				pos = _hierarchy_pos(
					G,
					child,
					width = dx,
					vert_gap = vert_gap,
					vert_loc = vert_loc-vert_gap, xcenter=nextx,
					pos=pos, parent = root
				)
		return pos


	return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)


def draw_graph(root_node, depth=None, top=None, visited=False, state_repository=None):
	def get_node_label(node):
		if node.parent is None:
			return "Root"

		if node.node_type == 0:
			total_value = f"\n{node.get_total_value(): .4f}"
			instant_value = f"\n{node.instant_value: .4f}"
			delta = f"\n{format(Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND[node.parent.children.index(node)] - 1, '.1e')}"
			label = f"{total_value}{instant_value}{delta}"
			if state_repository is not None:
				current_price = state_repository.retrieve(node.id).market_state.get_current_price('AUD', 'USD')
				# previous_price = state_repository.retrieve(node.parent.id).market_state.get_current_price('AUD', 'USD')
				label += f"\n{current_price: .4f}"
				# label += f"\n{(current_price - previous_price): .4f}"

		action = node.action
		label = ""
		if action is None:
			label = "None"
		elif action.action == 0:
			label = "Sell"
		elif action.action == 1:
			label = "Buy"
		elif action.action == 2:
			label = "Close"

		return f"{label}\n{node.total_value: .4f}"

	def get_edge_label(node):
		if node.node_type == 0:
			return format(node.weight, '.2e')
		return ""

	def get_node_color(node):
		if node.node_type == 0:
			return "darkred"
		return "navy"
	# TODO: YOU ARE HERE: ANALYZE WHERE THE TOTAL VALUE OF NONE ACTIONS COME FROM. BECAUSE THEIR DIRECT CHILDREN SEEM
	#  TO HAVE AN INSTANT VALUE OF 0.

	if depth is None:
		depth = get_max_depth(root_node)

	graph = nx.Graph()
	nodes = get_nodes(root_node, depth=depth, top=top, visited=visited) + [root_node]
	print(f"About to draw {len(nodes)}")
	for i, node in enumerate(nodes):
		if node is not root_node:
			graph.add_edge(i, nodes.index(node.parent))
	positions = hierarchy_pos(graph, nodes.index(root_node))
	nx.draw_networkx_labels(
		graph,
		pos=positions,
		labels={i: str(f"{get_node_label(node)}") for i, node in enumerate(nodes) if node is not root_node},
		font_size=7,
		font_color="whitesmoke"
	)
	nx.draw_networkx_edge_labels(
		graph,
		pos=positions,
		edge_labels={(i, nodes.index(node.parent)): get_edge_label(node) for i, node in enumerate(nodes) if node is not root_node},
		font_size=7,
		font_color="black",
	)
	nx.draw(graph, pos=positions, node_size=2500, node_color="navy", edge_color="silver")
	plt.show()


def draw_graph_live(root_node, depth=None, top=None, visited=False, state_repository=None, uct_fn=None):
	plt.ion()  # Turn on interactive mode

	def get_node_label(node):

		def get_action_label(action):
			if isinstance(action, ActionSequence):
				return ", ".join([get_action_label(action) for action in action.actions])

			if action is None:
				label = "None"
			elif action.action == 0:
				label ="Sell"
			elif action.action == 1:
				label = "Buy"
			elif action.action == 2:
				label = "Close"

			if action is not None and action.stop_loss is not None:
				label = f"{label}\n(SL={action.stop_loss})"

			return label

		if node.node_type == 0:
			total_value = f"\n{node.get_total_value(): .4f}"
			instant_value = f"\n{node.instant_value: .4f}"
			label = f"{total_value}{instant_value}"
			if state_repository is not None:
				from core.agent.agents.dnn_transition_agent import TraderDNNTransitionAgent

				target_instrument = TraderDNNTransitionAgent._get_target_instrument(
					state_repository.retrieve(node.parent.parent.id),
					None,
					state_repository.retrieve(node.id)
				)

				current_price = state_repository.retrieve(node.id).market_state.get_current_price(*target_instrument)
				previous_price = state_repository.retrieve(node.id).market_state.get_state_of(*target_instrument)[-2]
				label += f"\n{format(current_price - previous_price, '.1e').replace('e-0', 'e-')}"
				label += f"\n{current_price: .4f}"
			return label

		action = node.action
		label = get_action_label(action)

		label = f"{label}\n{node.total_value: .4f}\n{node.visits}"
		if uct_fn is not None:
			label += f"\n{uct_fn(node): 4f}"
		return label

	def get_edge_label(node):
		if node.node_type == 0:
			return format(node.weight, '.2e')
		return ""

	def get_node_color(node):
		if node.node_type == 0:
			return "darkred"
		return "navy"

	if depth is None:
		depth = get_max_depth(root_node)

	graph = nx.Graph()
	nodes = get_nodes(root_node, depth=depth, top=top, visited=visited) + [root_node]
	print(f"About to draw {len(nodes)}")
	for i, node in enumerate(nodes):
		if node is not root_node:
			graph.add_edge(i, nodes.index(node.parent))
	positions = hierarchy_pos(graph, nodes.index(root_node))

	# Clear previous plot
	plt.clf()

	nx.draw_networkx_labels(
		graph,
		pos=positions,
		labels={i: str(f"{get_node_label(node)}") for i, node in enumerate(nodes) if node is not root_node},
		font_size=7,
		font_color="whitesmoke"
	)
	nx.draw_networkx_edge_labels(
		graph,
		pos=positions,
		edge_labels={(i, nodes.index(node.parent)): get_edge_label(node) for i, node in enumerate(nodes) if
					 node is not root_node},
		font_size=7,
		font_color="black",
	)
	nx.draw(graph, pos=positions, node_size=2500, node_color="navy", edge_color="silver")

	# Show plot
	plt.draw()
	plt.pause(0.1)  # Pause to allow for plot update

	plt.ioff()  # Turn off interactive mode after drawing


def load_node(filename: str):
	from lib.rl.agent.mca.node import Node
	from core.agent.concurrency.mc.data.serializer import TraderNodeSerializer

	serializer = TraderNodeSerializer()

	with open(filename, "r") as file:
		json_ = json.load(file)

	return serializer.deserialize(json_)


def load_repository(filename: str):
	# from core.agent.concurrency.mc.data.serializer import TradeStateSerializer
	from lib.utils.staterepository import SectionalDictStateRepository

	# serializer = TradeStateSerializer()

	repository = SectionalDictStateRepository(
		2,
		15,
		# serializer=serializer
	)

	repository.load(filename)

	return repository

def load_node_repo(filepath):

	node = load_node(os.path.join(filepath, "graph.json"))
	repo = load_repository(os.path.join(filepath, "states.json"))
	return node, repo


def load_and_draw_graph(filepath):
	plt.figure()
	node, repo = load_node_repo(filepath)
	draw_graph_live(node, visited=True, state_repository=repo)
	return node, repo


def load_and_draw_graphs(container, bounds: typing.Tuple[int, int] = None):

	nodes_repos = []

	files = sorted(os.listdir(container))
	if bounds is not None:
		files = files[bounds[0]:bounds[1]]

	for i, filename in enumerate(files):
		print(f"Plotting {filename}")
		path = os.path.join(container, filename)
		if not os.path.isdir(path):
			continue
		nodes_repos.append(load_and_draw_graph(path))
		plt.show()

	return nodes_repos
