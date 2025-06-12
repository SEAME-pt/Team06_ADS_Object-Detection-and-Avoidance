using UnityEngine;

public class ChildObjectSwitcher : MonoBehaviour
{
    [Header("Configurações")]
    public Camera mainCamera;
    [Range(0, 89)] public float minPitch = 0f;
    [Range(0, 89)] public float maxPitch = 30f;
    [Range(0, 360)] public float minYaw = 0f;
    [Range(0, 360)] public float maxYaw = 360f;
    
    [Header("Configurações de Look At")]
    public float lookAtSpeed = 2f;
    public float distanceFromTarget = 5f;
    
    private Transform[] childObjects;
    private int currentChildIndex = 0;
    private bool isLookingAt = false;
    private Vector3 targetPosition;
    private Quaternion targetRotation;
    
    void Start()
    {
        // Se não foi atribuída uma camera, usa a main camera
        if (mainCamera == null)
            mainCamera = Camera.main;
            
        // Obtém todos os child objects
        childObjects = new Transform[transform.childCount];
        for (int i = 0; i < transform.childCount; i++)
        {
            childObjects[i] = transform.GetChild(i);
        }
        
        // Mostra apenas o primeiro objeto
        ShowCurrentChild();
    }
    
    void Update()
    {
        // Tecla I - próximo objeto
        if (Input.GetKeyDown(KeyCode.I))
        {
            NextChild();
        }
        
        // Tecla P - objeto anterior
        if (Input.GetKeyDown(KeyCode.P))
        {
            PreviousChild();
        }
        
        // Tecla O - Look At com rotação aleatória
        if (Input.GetKeyDown(KeyCode.O))
        {
            LookAtCurrentChild();
        }
        
        // Smooth look at animation
        if (isLookingAt)
        {
            mainCamera.transform.position = Vector3.Lerp(mainCamera.transform.position, targetPosition, Time.deltaTime * lookAtSpeed);
            mainCamera.transform.rotation = Quaternion.Lerp(mainCamera.transform.rotation, targetRotation, Time.deltaTime * lookAtSpeed);
            
            // Para a animação quando chegamos perto o suficiente
            if (Vector3.Distance(mainCamera.transform.position, targetPosition) < 0.1f &&
                Quaternion.Angle(mainCamera.transform.rotation, targetRotation) < 1f)
            {
                isLookingAt = false;
            }
        }
    }
    
    void ShowCurrentChild()
    {
        if (childObjects.Length == 0) return;
        
        // Esconde todos os child objects
        for (int i = 0; i < childObjects.Length; i++)
        {
            childObjects[i].gameObject.SetActive(false);
        }
        
        // Mostra apenas o atual
        childObjects[currentChildIndex].gameObject.SetActive(true);
        
        Debug.Log($"Mostrando: {childObjects[currentChildIndex].name} (Index: {currentChildIndex})");
    }
    
    void NextChild()
    {
        if (childObjects.Length == 0) return;
        
        currentChildIndex = (currentChildIndex + 1) % childObjects.Length;
        ShowCurrentChild();
    }
    
    void PreviousChild()
    {
        if (childObjects.Length == 0) return;
        
        currentChildIndex = (currentChildIndex - 1 + childObjects.Length) % childObjects.Length;
        ShowCurrentChild();
    }
    
    void LookAtCurrentChild()
    {
        if (childObjects.Length == 0 || mainCamera == null) return;
        
        Transform currentChild = childObjects[currentChildIndex];
        
        // Gera valores aleatórios para yaw e pitch
        float randomYaw = Random.Range(minYaw, maxYaw);
        float randomPitch = Random.Range(minPitch, maxPitch);
        
        // Converte para radianos
        float yawRad = randomYaw * Mathf.Deg2Rad;
        float pitchRad = randomPitch * Mathf.Deg2Rad;
        
        // Calcula a posição da camera baseada nos ângulos aleatórios
        Vector3 direction = new Vector3(
            Mathf.Sin(yawRad) * Mathf.Cos(pitchRad),
            Mathf.Sin(pitchRad),
            Mathf.Cos(yawRad) * Mathf.Cos(pitchRad)
        );
        
        // Define a posição alvo da camera
        targetPosition = currentChild.position - direction * distanceFromTarget;
        
        // Calcula a rotação para olhar para o objeto
        Vector3 lookDirection = (currentChild.position - targetPosition).normalized;
        targetRotation = Quaternion.LookRotation(lookDirection);
        
        // Inicia a animação smooth
        isLookingAt = true;
        
        Debug.Log($"Look At: {currentChild.name} - Yaw: {randomYaw:F1}°, Pitch: {randomPitch:F1}°");
    }
    
    // Métodos públicos para chamadas externas
    public void SetCurrentChild(int index)
    {
        if (index >= 0 && index < childObjects.Length)
        {
            currentChildIndex = index;
            ShowCurrentChild();
        }
    }
    
    public Transform GetCurrentChild()
    {
        if (childObjects.Length > 0)
            return childObjects[currentChildIndex];
        return null;
    }
    
    public int GetCurrentChildIndex()
    {
        return currentChildIndex;
    }
    
    public int GetChildCount()
    {
        return childObjects.Length;
    }
}
