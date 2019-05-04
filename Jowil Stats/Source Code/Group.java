package Jowil;

import com.jfoenix.controls.datamodels.treetable.RecursiveTreeObject;
import javafx.beans.property.SimpleStringProperty;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;

public class Group extends RecursiveTreeObject<Group> {

    //fields
    final private String name;
    final private Integer qCount;
    final private Integer realQCount; //number of questions including those blanked in answers keys
    final private SimpleStringProperty nameProp;
    final private SimpleStringProperty qCountProp;
    private ArrayList<String> possibleAnswers;
    private boolean isNumeric;
    private HashSet<String>correctAnswers;//set of all correct choices in the group



    Group(String name, int qCount,int realQCount){
        this.name=name;
        this.qCount=qCount;
        this.realQCount=realQCount;
        nameProp=new SimpleStringProperty(name);
        qCountProp=new SimpleStringProperty(Integer.toString(qCount));
        correctAnswers=new HashSet<>();
    }

    Group(Group original){
        this.name=original.name;
        this.qCount=original.qCount;
        this.realQCount=original.realQCount;
        this.nameProp=original.nameProp;
        this.qCountProp=original.qCountProp;
        this.isNumeric=original.isNumeric;
        this.possibleAnswers=new ArrayList<String>();
        for(String answer:original.possibleAnswers)
            this.possibleAnswers.add(answer);

        this.correctAnswers=(HashSet<String>)original.correctAnswers.clone();
    }


    //setters and getters
    public String getName() {
        return name;
    }

    public String getCleanedName(){
        return name.charAt(name.length()-1)=='-'?name.trim().substring(0,name.length()-1):name;
    }

    public void setNumeric(boolean numeric) {
        isNumeric = numeric;
    }


    public Integer getqCount() {
        return qCount;
    }



    public String getNameProp() {
        return nameProp.get();
    }

    public SimpleStringProperty getCleanedNameProp() {
        return new SimpleStringProperty(getCleanedName());
    }



    public SimpleStringProperty namePropProperty() {
        return nameProp;
    }


    public String getqCountProp() {
        return qCountProp.get();
    }

    public SimpleStringProperty getObsQcountProp(){
        return new SimpleStringProperty(getqCountProp());
    }

    public SimpleStringProperty qCountPropProperty() {
        return qCountProp;
    }



    public ArrayList<String> getPossibleAnswers() {
        return possibleAnswers;
    }

    public HashSet<String> getCorrectAnswers() {
        return correctAnswers;
    }

    public Integer getRealQCount() {
        return realQCount;
    }

    public void setCorrectAnswers(HashSet<String> correctAnswers) {
        this.correctAnswers = correctAnswers;
    }

    public void setPossibleAnswers(ArrayList<String> possibleAnswers) {
        this.possibleAnswers = possibleAnswers;
    }

    public boolean isBinaryResponse() {
        return possibleAnswers.get(0).toLowerCase().equals("t");
    }



    public void generatePossibleAnswers(String groupMaxChoice){

        isNumeric=true;
        possibleAnswers=new ArrayList<String>();
        int maxInt=0,minInt=1;
        String maxString="";
        char maxChar,minChar;
        try {
             maxInt = Integer.parseInt(groupMaxChoice);
        }catch(NumberFormatException e){
            isNumeric=false;
            maxChar=groupMaxChoice.charAt(0);//ignore extra letters
            if(Character.isLowerCase(new Character(maxChar)))
                minChar='a';
            else
                minChar='A';
            maxInt=maxChar;
            minInt=minChar;
        }



        for(int i=minInt;i<=maxInt;i++){
            String answer=isNumeric?Integer.toString(i):""+(char)i;
            possibleAnswers.add(answer);
        }


    }

    public void addAnswerToEnd(){

        String lastAnswer=possibleAnswers.get(possibleAnswers.size()-1);


        String newAnswer=isNumeric?Integer.toString(Integer.parseInt(lastAnswer)+1):""+(char)(lastAnswer.charAt(0)+1);

        possibleAnswers.add(newAnswer);


    }

    //returns null on success, otherwise returns the string that cannot be removed.
    public String updatePossibleAnswers(ArrayList<Boolean> isPossible, boolean ignoreCorrectAnswers){

        ArrayList<String> newPossibleAnswers=new ArrayList<String>();

        for(int i=0;i<possibleAnswers.size();i++){

            if(isPossible.get(i))
                newPossibleAnswers.add(possibleAnswers.get(i));
            else if(correctAnswers.contains(possibleAnswers.get(i)) && !ignoreCorrectAnswers)
                return possibleAnswers.get(i);

        }

        possibleAnswers=newPossibleAnswers;
        return null;
    }



}
