package controllers;

import dungeon.play.GameCharacter;
import dungeon.play.PlayMap;
import util.math2d.Point2D;
import util.statics.RandomNumberManager;

import java.util.ArrayList;
import java.util.List;

/**
 * Simple Monte Carlo Tree Search controller. It clones the current PlayMap,
 * simulates random rollouts up to a limited depth and picks the action with
 * the highest expected value (survival + proximity to exit).
 */
public class MCTSController extends Controller {
    private final int iterations;
    private final int rolloutDepth;
    private final double exploration;

    public MCTSController(PlayMap map, GameCharacter controllingChar) {
        this(map, controllingChar, 250, 8, Math.sqrt(2));
    }

    public MCTSController(PlayMap map, GameCharacter controllingChar, int iterations, int rolloutDepth, double exploration) {
        super(map, controllingChar, "MCTSController");
        this.iterations = iterations;
        this.rolloutDepth = rolloutDepth;
        this.exploration = exploration;
    }

    @Override
    public int getNextAction() {
        if (map == null || map.getHero() == null) {
            return PlayMap.IDLE;
        }
        return mctsSearch();
    }

    private int mctsSearch() {
        Node root = new Node(map.clone(), null, PlayMap.IDLE);

        for (int i = 0; i < iterations; i++) {
            Node node = select(root);
            if (!node.isTerminal()) {
                node = expand(node);
            }
            double reward = simulate(node.state);
            backpropagate(node, reward);
        }

        Node bestChild = root.bestChild(0); // exploitation only for final pick
        if (bestChild != null) {
            return bestChild.actionFromParent;
        }

        // Fallback: pick any valid move
        List<Integer> valid = getValidMoves(map);
        return valid.isEmpty() ? PlayMap.IDLE : valid.get(0);
    }

    private Node select(Node node) {
        while (!node.isTerminal() && node.isFullyExpanded()) {
            node = node.bestChild(exploration);
            if (node == null) {
                break;
            }
        }
        return node;
    }

    private Node expand(Node node) {
        int action = node.getUntriedAction();
        if (action == PlayMap.IDLE) {
            return node;
        }
        PlayMap nextState = cloneAndStep(node.state, action);
        Node child = new Node(nextState, node, action);
        node.children.add(child);
        return child;
    }

    private double simulate(PlayMap startState) {
        PlayMap simState = startState.clone();
        if (simState.getHero() == null) {
            return -1000;
        }
        for (int depth = 0; depth < rolloutDepth; depth++) {
            if (simState.isGameHalted() || simState.getHero() == null || !simState.getHero().isAlive()) {
                break;
            }
            List<Integer> moves = getValidMoves(simState);
            if (moves.isEmpty()) {
                break;
            }
            int move = moves.get(RandomNumberManager.getRandomInt(0, moves.size()));
            simState.updateGame(move);
        }
        return evaluate(simState);
    }

    private void backpropagate(Node node, double reward) {
        Node current = node;
        while (current != null) {
            current.visits++;
            current.value += reward;
            current = current.parent;
        }
    }

    private PlayMap cloneAndStep(PlayMap state, int action) {
        PlayMap clone = state.clone();
        clone.updateGame(action);
        return clone;
    }

    private List<Integer> getValidMoves(PlayMap state) {
        List<Integer> result = new ArrayList<Integer>();
        for (int i = 0; i < 4; i++) {
            Point2D next = state.getHero().getNextPosition(i);
            if (state.isValidMove(next)) {
                result.add(i);
            }
        }
        return result;
    }

    private double evaluate(PlayMap state) {
        if (state.getHero() == null) {
            return -1000;
        }
        if (!state.getHero().isAlive()) {
            return -1000;
        }
        double score = 0;
        if (state.isGameHalted()) {
            score += 500; // reached exit
        }
        Point2D exit = getTargetExit(state);
        double dist = state.getPaths().getDistance(state.getHero().getPosition(), exit);
        if (Double.isNaN(dist)) {
            dist = state.getMapSizeX() + state.getMapSizeY();
        }
        score -= dist;
        score += state.getHero().getScore() * 10;
        score += state.getHero().getHitpoints() * 0.5;
        return score;
    }

    private Point2D getTargetExit(PlayMap state) {
        int exitIndex = Math.min(1, Math.max(0, state.getExitLength() - 1));
        return state.getExit(exitIndex);
    }

    private static class Node {
        PlayMap state;
        Node parent;
        List<Node> children;
        int actionFromParent;
        double value;
        int visits;
        List<Integer> untriedActions;

        Node(PlayMap state, Node parent, int actionFromParent) {
            this.state = state;
            this.parent = parent;
            this.actionFromParent = actionFromParent;
            this.children = new ArrayList<Node>();
            this.untriedActions = new ArrayList<Integer>();
            this.untriedActions.addAll(getActions(state));
            this.value = 0;
            this.visits = 0;
        }

        boolean isTerminal() {
            return state.isGameHalted() || state.getHero() == null || !state.getHero().isAlive();
        }

        boolean isFullyExpanded() {
            return untriedActions.isEmpty();
        }

        int getUntriedAction() {
            if (untriedActions.isEmpty()) {
                return PlayMap.IDLE;
            }
            int index = RandomNumberManager.getRandomInt(0, untriedActions.size());
            return untriedActions.remove(index);
        }

        Node bestChild(double c) {
            Node best = null;
            double bestScore = Double.NEGATIVE_INFINITY;
            for (Node child : children) {
                if (child.visits == 0) {
                    return child; // force exploration of unvisited child
                }
                double exploitation = child.value / (double) child.visits;
                double exploration = c * Math.sqrt(Math.log(this.visits + 1) / (double) child.visits);
                double score = exploitation + exploration;
                if (score > bestScore) {
                    bestScore = score;
                    best = child;
                }
            }
            return best;
        }

        private static List<Integer> getActions(PlayMap state) {
            List<Integer> result = new ArrayList<Integer>();
            if (state.getHero() == null) {
                return result;
            }
            for (int i = 0; i < 4; i++) {
                Point2D next = state.getHero().getNextPosition(i);
                if (state.isValidMove(next)) {
                    result.add(i);
                }
            }
            return result;
        }
    }
}
