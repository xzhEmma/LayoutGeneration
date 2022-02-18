class Solution {
    public int countPoints(String rings) {
        HashMap<Integer,String> site = new HashMap< String>();
        for(int i=0;i<rings.length();i++){
            String color = rings.substring(2*i);
            String num = rings.substring(2*i+1);
            site.putIfAbsent(num,color);        
            
        }
        for(int i=0;i<site.size();i++){
            site.get(num);
        }
        
        

    }
}