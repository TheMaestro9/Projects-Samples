package Jowil;

import javafx.scene.control.Tooltip;

import static Jowil.Controller.isTranslationMode;
import static Jowil.Controller.translations;

public class TranslatableTooltip extends Tooltip {


    TranslatableTooltip(String text){
        super(isTranslationMode && translations.containsKey(text)?translations.get(text):text);
    }


}
