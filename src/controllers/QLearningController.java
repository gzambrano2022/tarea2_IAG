package controllers;

import dungeon.play.GameCharacter;
import dungeon.play.PlayMap;
import dungeon.play.Monster;
import dungeon.play.Powerup;
import dungeon.play.Reward;
import util.math2d.Point2D;

import java.io.*;
import java.util.*;

public class QLearningController extends Controller {
    private Random rng = new Random();

    // Hiperparámetros
    private double alpha = 0.15;
    private double gamma = 0.95;
    private static final int NUM_ACTIONS = 4;

    // Q-table persistente
    private static HashMap<String, double[]> qtable = null;
    private static double epsilon = 1.0;
    private static final double EPSILON_DECAY = 0.998;
    private static final double EPSILON_MIN = 0.05;
    private static int episodeCount = 0;
    private static boolean initialized = false;

    // Estado del episodio actual
    private String prevState = null;
    private int prevAction = -1;
    private int prevHP = 0;
    private int prevScore = 0;
    private double prevDistance = Double.MAX_VALUE;

    // Configuración de guardado
    private static final String SAVE_FILE = "testResults/qtable.ser";
    private static final int SAVE_INTERVAL = 50; // Guardar cada 50 episodios

    // Rango de visión para detectar entidades
    private static final int VISION_RANGE = 3;

    public QLearningController(PlayMap map, GameCharacter controllingChar){
        super(map, controllingChar, "QLearningController");

        if(!initialized) {
            loadQtable();
            initialized = true;
        }

        episodeCount++;

        // Inicializar estado previo
        if(controllingChar != null) {
            prevHP = controllingChar.getHitpoints();
        }
        if(map != null && map.getHero() != null) {
            prevScore = map.getHero().getScore();
            prevDistance = getDistanceToExit();
        }
    }

    @Override
    public int getNextAction(){
        String state = buildState();

        // Epsilon-greedy
        int action;
        if(rng.nextDouble() < epsilon){
            action = getRandomValidAction();
        } else {
            action = bestAction(state);
        }

        // Update Q-value del paso anterior
        if(prevState != null && prevAction != -1){
            double reward = computeReward();
            updateQ(prevState, prevAction, reward, state);
        }

        // Guardar estado actual
        prevState = state;
        prevAction = action;
        prevHP = controllingChar.getHitpoints();
        prevScore = map.getHero().getScore();
        prevDistance = getDistanceToExit();

        return action;
    }

    // Representacion de estado mejorada
    private String buildState(){
        Point2D hero = controllingChar.getPosition();
        int heroX = (int)hero.x;
        int heroY = (int)hero.y;
        Point2D exit = map.getExit(1);

        StringBuilder state = new StringBuilder();

        // Categoría de HP
        int hpCategory = getHPCategory();
        state.append("HP").append(hpCategory).append("|");

        // Dirección hacia la salida
        int dx = (int)(exit.x - hero.x);
        int dy = (int)(exit.y - hero.y);
        String dirExit = getDirectionCode(dx, dy);
        state.append("EXIT").append(dirExit).append("|");

        // Distancia a la salida
        double distToExit = Math.sqrt(dx*dx + dy*dy);
        int distCategory = getDistanceCategory(distToExit);
        state.append("DIST").append(distCategory).append("|");

        // Monstruos cercanos
        String monsterInfo = getNearbyMonsters(heroX, heroY);
        state.append("MON").append(monsterInfo).append("|");

        // Pociones cercanas
        if(hpCategory <= 2) { // Solo si HP bajo/medio
            String potionInfo = getNearbyPotions(heroX, heroY);
            state.append("POT").append(potionInfo).append("|");
        }

        // Recompensas cercanas
        int nearbyRewards = countNearbyRewards(heroX, heroY);
        state.append("REW").append(nearbyRewards).append("|");

        // Configuración de paredes
        state.append("W");
        for(int dir = 0; dir < 4; dir++){
            Point2D nextPos = controllingChar.getNextPosition(dir);
            state.append(map.isValidMove(nextPos) ? "0" : "1");
        }

        return state.toString();
    }

    // Categoriza HP en 5 niveles
    private int getHPCategory(){
        int hp = controllingChar.getHitpoints();
        int maxHP = map.getHero().getStartingHitpoints();

        double ratio = (double)hp / maxHP;
        if(ratio > 0.8) return 4;  // Excelente
        if(ratio > 0.6) return 3;  // Bueno
        if(ratio > 0.4) return 2;  // Medio
        if(ratio > 0.2) return 1;  // Bajo
        return 0;                   // Crítico
    }


    // Obtiene código de dirección
    private String getDirectionCode(int dx, int dy){
        if(Math.abs(dx) < 2 && Math.abs(dy) < 2) return "HERE";

        // Normalizar a 8 direcciones
        if(Math.abs(dx) > Math.abs(dy) * 2){
            return dx > 0 ? "E" : "W";
        } else if(Math.abs(dy) > Math.abs(dx) * 2){
            return dy > 0 ? "S" : "N";
        } else {
            if(dx > 0 && dy > 0) return "SE";
            if(dx > 0 && dy < 0) return "NE";
            if(dx < 0 && dy > 0) return "SW";
            if(dx < 0 && dy < 0) return "NW";
        }
        return "HERE";
    }


    // Categoriza distancia
    private int getDistanceCategory(double dist){
        if(dist < 3) return 0;
        if(dist < 6) return 1;
        if(dist < 10) return 2;
        if(dist < 15) return 3;
        return 4;
    }


    // Detecta monstruos cercanos y su dirección
    private String getNearbyMonsters(int heroX, int heroY){
        List<String> threats = new ArrayList<>();

        for(Monster monster : map.getMonsterChars()){
            if(!monster.isAlive()) continue;

            Point2D monsterPos = monster.getPosition();
            int dx = (int)(monsterPos.x - heroX);
            int dy = (int)(monsterPos.y - heroY);
            double dist = Math.sqrt(dx*dx + dy*dy);

            if(dist <= VISION_RANGE){
                String dir = getDirectionCode(dx, dy);
                int damage = monster.getDamage();

                // Categorizar amenaza por distancia
                if(dist <= 1.5){
                    threats.add(dir + "!" + (damage/5)); // Muy cerca
                } else {
                    threats.add(dir + "-" + (damage/5)); // Cerca
                }
            }
        }

        if(threats.isEmpty()) return "SAFE";

        Collections.sort(threats);
        return String.join(",", threats);
    }

    // Detecta pociones cercanas
    private String getNearbyPotions(int heroX, int heroY){
        String closest = "NONE";
        double minDist = Double.MAX_VALUE;

        for(Powerup potion : map.getPotionChars()){
            if(!potion.isAlive()) continue;

            Point2D potionPos = potion.getPosition();
            int dx = (int)(potionPos.x - heroX);
            int dy = (int)(potionPos.y - heroY);
            double dist = Math.sqrt(dx*dx + dy*dy);

            if(dist <= VISION_RANGE && dist < minDist){
                minDist = dist;
                closest = getDirectionCode(dx, dy);
            }
        }

        return closest;
    }


    //Cuenta recompensas cercanas
    private int countNearbyRewards(int heroX, int heroY){
        int count = 0;

        for(Reward reward : map.getRewardChars()){
            if(!reward.isAlive()) continue;

            Point2D rewardPos = reward.getPosition();
            int dx = (int)(rewardPos.x - heroX);
            int dy = (int)(rewardPos.y - heroY);
            double dist = Math.sqrt(dx*dx + dy*dy);

            if(dist <= VISION_RANGE) count++;
        }

        return count;
    }

    // Sistema de recompensas
    private double computeReward(){
        double reward = 0;

        // 1. Penalización base por paso
        reward -= 0.1;

        // 2. Cambios en HP
        int currentHP = controllingChar.getHitpoints();
        int hpDiff = currentHP - prevHP;

        if(hpDiff < 0){
            // Perdió HP: penalización severa
            reward += hpDiff * 3.0;

            // Penalización extra si HP está bajo
            if(currentHP < 15){
                reward -= 5.0;
            }
        } else if(hpDiff > 0){
            // Ganó HP: recompensa moderada
            reward += hpDiff * 1.5;
        }

        // Cambios en score
        int currentScore = map.getHero().getScore();
        int scoreDiff = currentScore - prevScore;
        if(scoreDiff > 0){
            reward += scoreDiff * 10.0;
        }

        // Progreso hacia la salida
        double currentDist = getDistanceToExit();
        if(prevDistance != Double.MAX_VALUE){
            double distDiff = prevDistance - currentDist;

            if(distDiff > 0){
                // Se acercó a la salida
                reward += 0.5;
            } else if(distDiff < 0){
                // Se alejó
                reward -= 0.2;
            }
        }

        // Muerte: penalización ENORME
        if(currentHP <= 0){
            reward -= 150.0;
        }

        // Victoria: recompensa ENORME
        Point2D hero = controllingChar.getPosition();
        Point2D exit = map.getExit(1);
        if(hero.isAt(exit)){
            reward += 200.0;

            // Bonus por HP restante
            reward += currentHP * 0.5;
        }

        return reward;
    }

    private double getDistanceToExit(){
        Point2D hero = controllingChar.getPosition();
        Point2D exit = map.getExit(1);

        // Usar distancia Manhattan
        return Math.abs(exit.x - hero.x) + Math.abs(exit.y - hero.y);
    }

    // Logica qlearning
    private double[] getQ(String s){
        return qtable.computeIfAbsent(s, k -> new double[NUM_ACTIONS]);
    }

    private int bestAction(String s){
        double[] qs = getQ(s);
        double best = Double.NEGATIVE_INFINITY;
        List<Integer> bestActions = new ArrayList<>();

        // Solo considerar acciones válidas
        for(int i = 0; i < NUM_ACTIONS; i++){
            Point2D nextPos = controllingChar.getNextPosition(i);
            if(!map.isValidMove(nextPos)) continue;

            if(qs[i] > best){
                best = qs[i];
                bestActions.clear();
                bestActions.add(i);
            } else if(qs[i] == best){
                bestActions.add(i);
            }
        }

        if(bestActions.isEmpty()){
            return getRandomValidAction();
        }

        return bestActions.get(rng.nextInt(bestActions.size()));
    }

    private void updateQ(String s, int a, double reward, String s2){
        double[] qOld = getQ(s);
        double[] qNew = getQ(s2);

        double maxNext = Double.NEGATIVE_INFINITY;
        for(int i = 0; i < NUM_ACTIONS; i++){
            if(qNew[i] > maxNext){
                maxNext = qNew[i];
            }
        }
        if(maxNext == Double.NEGATIVE_INFINITY) maxNext = 0;

        // Q-Learning: Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
        qOld[a] = qOld[a] + alpha * (reward + gamma * maxNext - qOld[a]);
    }

    private int getRandomValidAction(){
        List<Integer> validActions = new ArrayList<>();

        for(int i = 0; i < NUM_ACTIONS; i++){
            Point2D nextPos = controllingChar.getNextPosition(i);
            if(map.isValidMove(nextPos)){
                validActions.add(i);
            }
        }

        if(validActions.isEmpty()){
            return PlayMap.IDLE;
        }

        return validActions.get(rng.nextInt(validActions.size()));
    }

    // Persistencia optimizada
    private void loadQtable(){
        try{
            new File("testResults").mkdirs();
            File file = new File(SAVE_FILE);

            if(!file.exists()){
                System.out.println("[Q-Learning] Iniciando Q-table nueva.");
                qtable = new HashMap<>();
                return;
            }

            ObjectInputStream ois = new ObjectInputStream(new FileInputStream(SAVE_FILE));
            qtable = (HashMap<String, double[]>)ois.readObject();
            ois.close();

            System.out.println("[Q-Learning] Q-table cargada: " + qtable.size() + " estados.");
        } catch(Exception e){
            System.out.println("[Q-Learning] Error cargando Q-table: " + e.getMessage());
            qtable = new HashMap<>();
        }
    }

    private void saveQtable(){
        try{
            new File("testResults").mkdirs();
            ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(SAVE_FILE));
            oos.writeObject(qtable);
            oos.close();

            System.out.printf("[Q-Learning] Episodio %d | Q-table: %d estados | Epsilon: %.3f%n",
                    episodeCount, qtable.size(), epsilon);
        } catch(Exception e){
            System.err.println("[Q-Learning] Error guardando Q-table: " + e.getMessage());
        }
    }

    @Override
    public void reset(){
        // Actualizar Q-value final si terminó el episodio
        if(prevState != null && prevAction != -1){
            double finalReward = computeReward();

            // Estado terminal: no hay siguiente estado
            double[] qOld = getQ(prevState);
            qOld[prevAction] = qOld[prevAction] + alpha * (finalReward - qOld[prevAction]);
        }

        // Guardar Q-table periódicamente
        if(episodeCount % SAVE_INTERVAL == 0){
            saveQtable();
        }

        // Epsilon decay
        epsilon = Math.max(EPSILON_MIN, epsilon * EPSILON_DECAY);

        // Resetear variables del episodio
        prevState = null;
        prevAction = -1;
        prevHP = 0;
        prevScore = 0;
        prevDistance = Double.MAX_VALUE;
    }

    // getters para monitoreo
    public static double getEpsilon(){ return epsilon; }
    public static int getEpisodeCount(){ return episodeCount; }
    public static int getQTableSize(){ return qtable != null ? qtable.size() : 0; }
    public static void setEpsilon(double e){ epsilon = Math.max(EPSILON_MIN, Math.min(1.0, e)); }
    public static void setAlpha(double a){ /* alpha no es estático */ }


    // Forzar guardado manual de Q-table
    public void forceSave(){
        saveQtable();
    }
}