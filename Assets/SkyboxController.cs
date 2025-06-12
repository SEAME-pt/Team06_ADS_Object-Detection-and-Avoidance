using UnityEngine;

public class SkyboxController : MonoBehaviour
{
    [Header("Skybox Materials")]
    public Material[] skyboxMaterials;
    
    [Header("Configurações")]
    [Tooltip("Velocidade de transição entre skyboxes (0 = instantâneo)")]
    public float transitionSpeed = 1f;
    
    private int currentSkyboxIndex = 0;
    private bool isTransitioning = false;
    private float transitionTimer = 0f;
    private Material previousSkybox;
    private Material targetSkybox;
    
    void Start()
    {
        // Define o primeiro skybox se existir
        if (skyboxMaterials.Length > 0)
        {
            SetCurrentSkybox();
        }
        else
        {
            Debug.LogWarning("Nenhum material de skybox foi atribuído!");
        }
    }
    
    void Update()
    {
        // Tecla N - próximo skybox
        if (Input.GetKeyDown(KeyCode.N))
        {
            NextSkybox();
        }
        
        // Tecla M - skybox anterior
        if (Input.GetKeyDown(KeyCode.M))
        {
            PreviousSkybox();
        }
        
        // Transição suave entre skyboxes
        if (isTransitioning && transitionSpeed > 0)
        {
            transitionTimer += Time.deltaTime * transitionSpeed;
            
            if (transitionTimer >= 1f)
            {
                // Transição completa
                RenderSettings.skybox = targetSkybox;
                isTransitioning = false;
                transitionTimer = 0f;
            }
            else
            {
                // Interpola entre os skyboxes (se suportado pelo material)
                UpdateSkyboxTransition();
            }
        }
    }
    
    void SetCurrentSkybox()
    {
        if (skyboxMaterials.Length == 0) return;
        
        if (transitionSpeed > 0 && RenderSettings.skybox != null)
        {
            // Inicia transição suave
            previousSkybox = RenderSettings.skybox;
            targetSkybox = skyboxMaterials[currentSkyboxIndex];
            isTransitioning = true;
            transitionTimer = 0f;
        }
        else
        {
            // Mudança instantânea
            RenderSettings.skybox = skyboxMaterials[currentSkyboxIndex];
        }
        
        // Atualiza a iluminação ambiente
        DynamicGI.UpdateEnvironment();
        
        Debug.Log($"Skybox alterado para: {skyboxMaterials[currentSkyboxIndex].name} (Index: {currentSkyboxIndex})");
    }
    
    void NextSkybox()
    {
        if (skyboxMaterials.Length == 0) return;
        
        currentSkyboxIndex = (currentSkyboxIndex + 1) % skyboxMaterials.Length;
        SetCurrentSkybox();
    }
    
    void PreviousSkybox()
    {
        if (skyboxMaterials.Length == 0) return;
        
        currentSkyboxIndex = (currentSkyboxIndex - 1 + skyboxMaterials.Length) % skyboxMaterials.Length;
        SetCurrentSkybox();
    }
    
    void UpdateSkyboxTransition()
    {
        // Esta função pode ser expandida para criar transições mais complexas
        // Por exemplo, misturar dois skyboxes ou fazer fade
        
        // Para transições simples, apenas define o skybox alvo
        // Transições mais avançadas requerem shaders personalizados
        RenderSettings.skybox = targetSkybox;
    }
    
    // Métodos públicos para controlo externo
    public void SetSkybox(int index)
    {
        if (index >= 0 && index < skyboxMaterials.Length)
        {
            currentSkyboxIndex = index;
            SetCurrentSkybox();
        }
        else
        {
            Debug.LogWarning($"Índice de skybox inválido: {index}");
        }
    }
    
    public void SetSkyboxByName(string skyboxName)
    {
        for (int i = 0; i < skyboxMaterials.Length; i++)
        {
            if (skyboxMaterials[i].name == skyboxName)
            {
                currentSkyboxIndex = i;
                SetCurrentSkybox();
                return;
            }
        }
        Debug.LogWarning($"Skybox com nome '{skyboxName}' não encontrado!");
    }
    
    public Material GetCurrentSkybox()
    {
        if (skyboxMaterials.Length > 0 && currentSkyboxIndex < skyboxMaterials.Length)
            return skyboxMaterials[currentSkyboxIndex];
        return null;
    }
    
    public int GetCurrentSkyboxIndex()
    {
        return currentSkyboxIndex;
    }
    
    public int GetSkyboxCount()
    {
        return skyboxMaterials.Length;
    }
    
    public string[] GetSkyboxNames()
    {
        string[] names = new string[skyboxMaterials.Length];
        for (int i = 0; i < skyboxMaterials.Length; i++)
        {
            names[i] = skyboxMaterials[i] != null ? skyboxMaterials[i].name : "Null";
        }
        return names;
    }
  
}
