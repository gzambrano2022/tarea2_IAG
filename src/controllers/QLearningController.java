package controllers;

import dungeon.play.GameCharacter;
import dungeon.play.PlayMap;
import util.math2d.Point2D;

import java.io.*;
import java.util.*;

/**
 * Q-Learning Controller para MiniDungeons
 *
 * Representación del estado:
 *  dx_sign | dy_sign | wall_up | wall_right | wall_down | wall_left
 *
 * Acciones:
 *  0 = UP, 1 = RIGHT, 2 = DOWN, 3 = LEFT
 */
public class QLearningController extends Controller {
    private Random rng = new Random();

    // Hiperparámetros
    private double alpha = 0.3;      // learning rate
    private double gamma = 0.95;     // discount factor
    private int NUM_ACTIONS = 4;

    // CRÍTICO: Q-table ESTÁTICA para persistir entre instancias
    private static HashMap<String, double[]> qtable = null;

    // Epsilon estático para decay progresivo
    private static double epsilon = 0.3;
    private static final double EPSILON_DECAY = 0.995;
    private static final double EPSILON_MIN = 0.01;

    // Contador de episodios
    private static int episodeCount = 0;

    // Flag para saber si ya se inicializó
    private static boolean initialized = false;

    // Estado previo para update (no estático, específico de cada episodio)
    private String prevState = null;
    private int prevAction = -1;
    private double prevDistance = Double.MAX_VALUE;

    private static final String SAVE_FILE = "testResults/qtable.ser";

    public QLearningController(PlayMap map, GameCharacter controllingChar){
        super(map, controllingChar, "QLearningController");

        // Inicializar solo la primera vez
        if(!initialized) {
            loadQtable();
            initialized = true;
        }

        episodeCount++;
    }

    @Override
    public int getNextAction(){
        String state = buildState();

        int action;
        if(rng.nextDouble() < epsilon){
            action = rng.nextInt(NUM_ACTIONS);
        } else {
            action = bestAction(state);
        }

        // Update Q-value con el estado previo
        if(prevState != null && prevAction != -1){
            double reward = computeReward();
            updateQ(prevState, prevAction, reward, state);
        }

        prevState = state;
        prevAction = action;
        prevDistance = getDistanceToExit();

        return action;
    }

    // -----------------
    // STATE REPRESENTATION
    // -----------------
    private String buildState(){
        Point2D hero = controllingChar.getPosition();
        Point2D exit = map.getExit(1);

        // Dirección hacia la salida
        int dx = (int)(exit.x - hero.x);
        int dy = (int)(exit.y - hero.y);

        String dxs = (dx < 0) ? "L" : (dx > 0 ? "R" : "0");
        String dys = (dy < 0) ? "U" : (dy > 0 ? "D" : "0");

        // Paredes en las 4 direcciones
        int wallUp    = map.isValidMove(new Point2D(hero.x, hero.y - 1)) ? 0 : 1;
        int wallRight = map.isValidMove(new Point2D(hero.x + 1, hero.y)) ? 0 : 1;
        int wallDown  = map.isValidMove(new Point2D(hero.x, hero.y + 1)) ? 0 : 1;
        int wallLeft  = map.isValidMove(new Point2D(hero.x - 1, hero.y)) ? 0 : 1;

        return dxs + "|" + dys + "|" + wallUp + wallRight + wallDown + wallLeft;
    }

    // -----------------
    // REWARD FUNCTION
    // -----------------
    private double computeReward(){
        Point2D hero = controllingChar.getPosition();
        Point2D exit = map.getExit(1);

        // Recompensa grande por llegar a la salida
        if (hero.isAt(exit)) {
            return +100.0;
        }

        // Penalización por morir
        if (controllingChar.getHitpoints() <= 0) {
            return -50.0;
        }

        // Recompensa por acercarse a la salida
        double currentDistance = getDistanceToExit();
        double distanceReward = 0;

        if(prevDistance != Double.MAX_VALUE) {
            if(currentDistance < prevDistance) {
                distanceReward = +1.0; // Se acercó
            } else if(currentDistance > prevDistance) {
                distanceReward = -1.0; // Se alejó
            }
        }

        // Penalización por cada paso (incentiva rutas cortas)
        return distanceReward - 0.1;
    }

    private double getDistanceToExit() {
        Point2D hero = controllingChar.getPosition();
        Point2D exit = map.getExit(1);
        return Math.abs(exit.x - hero.x) + Math.abs(exit.y - hero.y);
    }

    // -----------------
    // Q-TABLE LOGIC
    // -----------------
    private double[] getQ(String s){
        return qtable.computeIfAbsent(s, k -> new double[NUM_ACTIONS]);
    }

    private int bestAction(String s){
        double[] qs = getQ(s);
        double best = Double.NEGATIVE_INFINITY;
        List<Integer> bestActions = new ArrayList<>();

        for (int i = 0; i < NUM_ACTIONS; i++) {
            if (qs[i] > best) {
                best = qs[i];
                bestActions.clear();
                bestActions.add(i);
            } else if (qs[i] == best) {
                bestActions.add(i);
            }
        }

        // Desempate aleatorio
        return bestActions.get(rng.nextInt(bestActions.size()));
    }

    private void updateQ(String s, int a, double reward, String s2){
        double[] qOld = getQ(s);
        double[] qNew = getQ(s2);

        double maxNext = Arrays.stream(qNew).max().orElse(0.0);

        // Fórmula de Q-Learning: Q(s,a) = Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
        qOld[a] = qOld[a] + alpha * (reward + gamma * maxNext - qOld[a]);
    }

    // -----------------
    // PERSISTENCE
    // -----------------
    private void loadQtable() {
        try {
            new File("testResults").mkdirs();

            File file = new File(SAVE_FILE);
            if(!file.exists()) {
                System.out.println("Iniciando Q-table nueva.");
                qtable = new HashMap<>();
                return;
            }

            ObjectInputStream ois = new ObjectInputStream(new FileInputStream(SAVE_FILE));
            qtable = (HashMap<String, double[]>) ois.readObject();
            ois.close();

            System.out.println("Q-table cargada: " + qtable.size() + " estados. Episodio " + episodeCount);
        } catch (Exception e) {
            System.out.println("Error cargando Q-table, iniciando nueva: " + e.getMessage());
            qtable = new HashMap<>();
        }
    }

    private void saveQtable() {
        try {
            new File("testResults").mkdirs();

            ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(SAVE_FILE));
            oos.writeObject(qtable);
            oos.close();

            System.out.println("Q-table guardada: " + qtable.size() + " estados. Epsilon: " + String.format("%.3f", epsilon));
        } catch (Exception e) {
            System.err.println("Error guardando Q-table: " + e.getMessage());
        }
    }

    @Override
    public void reset() {
        // Este método se llama al final de cada episodio
        saveQtable();

        // Epsilon decay después de cada episodio
        epsilon = Math.max(EPSILON_MIN, epsilon * EPSILON_DECAY);

        // Resetear variables del episodio
        prevState = null;
        prevAction = -1;
        prevDistance = Double.MAX_VALUE;
    }
}