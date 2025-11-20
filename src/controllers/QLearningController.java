package controllers;

import dungeon.play.GameCharacter;
import dungeon.play.PlayMap;
import util.math2d.Point2D;

import java.io.*;
import java.util.*;
import java.io.FileInputStream;
import java.io.ObjectInputStream;



/**
 * Aprende a moverse hacia la salida a través de espisodios.
 *
 * Representación del estado:
 *  dx_sign | dy_sign | wall_up | wall_right | wall_down | wall_left
 *
 *  Acciones:
 *  0 = UP
 *  1 = RIGHT
 *  2 = DOWN
 *  3 = LEFT
 */


public class QLearningController extends Controller {
    private Random rng = new Random();

    // Hiperparámetros
    private double alpha = 0.20; // learning rate
    private double gamma = 0.90; // discount factor
    private double epsilon = 0.25; // exploración (epsilon-greedy)
    private int NUM_ACTIONS = 4; // up, right, down, left

    // Q-table
    private HashMap<String, double[]> qtable = new HashMap<>();

    // Estado previo para update
    private String prevState = null;
    private int prevAction = -1;

    private final String SAVE_FILE = "./testResults/qtable.ser";

    public QLearningController(PlayMap map, GameCharacter controllingChar){
        super(map, controllingChar, "QLearningController");
        loadQtable();
    }

    @Override
    public int getNextAction(){
        // construir estado acutal
        String state = buildState();

        int action;
        // epsilon-greedy
        if(rng.nextDouble() < epsilon){
            action = rng.nextInt(NUM_ACTIONS); // explorar
        } else {
            action = BestAction(state); // explotar
        }

        // si tenemos un estado previo, actualizamso
        if(prevState != null && prevAction != -1){
            double reward = computeReward();
            updateQ(prevState, prevAction, reward, state);
        }

        // actualizar prev
        prevState = state;
        prevAction = action;

        return action;
    }

// -----------------
// STATE REPRESENTATION
// -----------------
    private String buildState(){
        Point2D hero = controllingChar.getPosition();
        Point2D exit = map.getExit(1);

        // diferencias discretizadas
        int dx = (int)(exit.x - hero.x);
        int dy = (int)(exit.y - hero.y);

        String dxs = (dx < 0) ? "L" : (dx>0 ? "R" : "0");
        String dys = (dy < 0) ? "U" : (dy>0 ? "D" : "0");

        // paredes alrededor
        int wallUp    = map.isValidMove(new Point2D(hero.x, hero.y - 1)) ? 0 : 1;
        int wallRight = map.isValidMove(new Point2D(hero.x + 1, hero.y)) ? 0 : 1;
        int wallDown  = map.isValidMove(new Point2D(hero.x, hero.y + 1)) ? 0 : 1;
        int wallLeft  = map.isValidMove(new Point2D(hero.x - 1, hero.y)) ? 0 : 1;

        return dxs + "|" + dys + "|" + wallUp + wallRight + wallDown + wallLeft;
    }

// -----------------
// REWARD
// -----------------

    private double computeReward(){
        Point2D hero = controllingChar.getPosition();
        Point2D exit = map.getExit(1);

        if (hero.isAt(exit)) {
            return +100;   // llegó a la meta
        }

        return -1; // paso normal
    }

// -----------------
// Q-TABLE LOGIC
// -----------------

    private double[] getQ(String s){
        return qtable.computeIfAbsent(s, k -> new double[NUM_ACTIONS]);
    }

    private int BestAction(String s){
        double[] qs = getQ(s);
        double best = Double.NEGATIVE_INFINITY;
        int bestA = 0;

        for (int i = 0; i < NUM_ACTIONS; i++) {
            if (qs[i] > best) {
                best = qs[i];
                bestA = i;
            }
        }
        return bestA;
    }

    private void updateQ(String s, int a, double reward, String s2){
        double[] qOld = getQ(s);
        double[] qNew = getQ(s2);

        double maxNext = Arrays.stream(qNew).max().orElse(0);

        qOld[a] = qOld[a] + alpha * (reward + gamma * maxNext - qOld[a]);
    }

// -----------------
// SAVE / LOAD
// -----------------

    private void loadQtable() {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(SAVE_FILE))) {
            qtable = (HashMap<String, double[]>) ois.readObject();
            System.out.println("Q-table cargada con " + qtable.size() + " estados.");
        } catch (Exception e) {
            System.out.println("No hay Q-table previa. Empezando de cero.");
            qtable = new HashMap<>();   // ← OBLIGATORIO
        }
    }

    private void saveQtable() {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(SAVE_FILE))) {
            oos.writeObject(qtable);
            System.out.println("Q-table guardada.");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public void reset() {
        saveQtable();
        if (qtable == null) qtable = new HashMap<>();
        prevState = null;
        prevAction = -1;
    }

}
