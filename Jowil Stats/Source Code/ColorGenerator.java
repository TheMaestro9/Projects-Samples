package Jowil;

import javafx.scene.paint.Color;

import java.util.HashSet;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.LinkedBlockingQueue;


public class ColorGenerator {
    private final static String [] manualModeTableColors={"#00CED1","#DC143C","#7FFFD4","#FF6347","#228B22","#FFD700","#FFC0CB","#FFA500","#EE82EE","#d9d9d9","#90EE90"};
    LinkedBlockingDeque<String> availableColors;
    HashSet history=new HashSet();
    int brighterCounter=0;

    ColorGenerator(){

        resetAvailable();
    }

    public String getNextColor(){

        String color=availableColors.poll();
        if(color==null){
            resetAvailableToDarker();
            color=availableColors.poll();
        }
        return color;
    }

    public void addToAvailable(String color){
        availableColors.addFirst(color);
        history.add(color);
    }


    public void resetAvailable(){

        availableColors=new LinkedBlockingDeque<>();
        for(String color:manualModeTableColors){
            availableColors.add(color);
            Color col=Color.web(color);
            double red=col.getRed()*255;
            double blue=col.getBlue()*255;
            double green=col.getGreen()*255;
            String colorRGB=String.format("rgb(%d,%d,%d)",(int)red,(int)blue,(int)green);
            history.add(colorRGB);
        }

    }


    private void resetAvailableToDarker(){
        for(String s:manualModeTableColors){
            Color color=Color.web(s);
            double red=color.getRed()*(0.8-0.1*brighterCounter)*255;
            double blue=color.getBlue()*(0.8-0.1*brighterCounter)*255;
            double green=color.getGreen()*(0.8-0.1*brighterCounter)*255;
            String newColor=String.format("rgb(%d,%d,%d)",(int)red,(int)blue,(int)green);

            if(history.contains(newColor)) //must be unique ->ignore new color
                continue;

            availableColors.add(newColor);
            history.add(newColor);
        }
        brighterCounter++;
    }
}
